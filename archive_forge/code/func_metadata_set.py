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
@resource_objects.retry_on_conflict
def metadata_set(self, metadata, merge_metadata=None):
    """Write new metadata to the database.

        The caller may optionally provide a merge_metadata() function, which
        takes two arguments - the metadata passed to metadata_set() and the
        current metadata of the resource - and returns the merged metadata to
        write. If merge_metadata is not provided, the metadata passed to
        metadata_set() is written verbatim, overwriting any existing metadata.

        If a race condition is detected, the write will be retried with the new
        result of merge_metadata() (if it is supplied) or the verbatim data (if
        it is not).
        """
    if self.id is None or self.action == self.INIT:
        raise exception.ResourceNotAvailable(resource_name=self.name)
    refresh = merge_metadata is not None
    db_res = resource_objects.Resource.get_obj(self.stack.context, self.id, refresh=refresh, fields=('name', 'rsrc_metadata', 'atomic_key', 'engine_id', 'action', 'status'))
    if db_res.action == self.DELETE:
        self._db_res_is_deleted = True
        LOG.debug('resource %(name)s, id: %(id)s is DELETE_%(st)s, not setting metadata', {'name': self.name, 'id': self.id, 'st': db_res.status})
        raise exception.ResourceNotAvailable(resource_name=self.name)
    LOG.debug('Setting metadata for %s', str(self))
    if refresh:
        metadata = merge_metadata(metadata, db_res.rsrc_metadata)
    if db_res.update_metadata(metadata):
        self._incr_atomic_key(db_res.atomic_key)
    self._rsrc_metadata = metadata