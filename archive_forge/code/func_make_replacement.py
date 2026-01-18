import collections
import contextlib
import datetime as dt
import itertools
import pydoc
import re
import tenacity
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import short_id
from heat.common import timeutils
from heat.engine import attributes
from heat.engine.cfn import template as cfn_tmpl
from heat.engine import clients
from heat.engine.clients import default_client_plugin
from heat.engine import environment
from heat.engine import event
from heat.engine import function
from heat.engine.hot import template as hot_tmpl
from heat.engine import node_data
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import status
from heat.engine import support
from heat.engine import sync_point
from heat.engine import template
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_objects
from heat.objects import resource_properties_data as rpd_objects
from heat.rpc import client as rpc_client
def make_replacement(self, new_tmpl_id, requires):
    """Create a replacement resource in the database.

        Returns the DB ID of the new resource, or None if the new resource
        cannot be created (generally because the template ID does not exist).
        Raises UpdateInProgress if another traversal has already locked the
        current resource.
        """
    rs = {'stack_id': self.stack.id, 'name': self.name, 'rsrc_prop_data_id': None, 'needed_by': [], 'requires': sorted(requires, reverse=True), 'replaces': self.id, 'action': self.INIT, 'status': self.COMPLETE, 'current_template_id': new_tmpl_id, 'stack_name': self.stack.name, 'root_stack_id': self.root_stack_id}
    attempts = max(cfg.CONF.client_retry_limit, 0) + 1

    def prepare_attempt(retry_state):
        if retry_state.attempt_number > 1:
            res_obj = resource_objects.Resource.get_obj(self.context, self.id)
            if res_obj.engine_id is not None or res_obj.updated_at != self.updated_time:
                raise exception.UpdateInProgress(resource_name=self.name)
            self._atomic_key = res_obj.atomic_key

    @tenacity.retry(stop=tenacity.stop_after_attempt(attempts), retry=tenacity.retry_if_exception_type(exception.UpdateInProgress), before=prepare_attempt, wait=tenacity.wait_random(max=2), reraise=True)
    def create_replacement():
        return resource_objects.Resource.replacement(self.context, self.id, rs, self._atomic_key)
    new_rs = create_replacement()
    if new_rs is None:
        return None
    self._incr_atomic_key(self._atomic_key)
    self.replaced_by = new_rs.id
    return new_rs.id