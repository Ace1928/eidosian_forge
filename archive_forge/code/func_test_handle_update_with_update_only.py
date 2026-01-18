import contextlib
import copy
import re
from unittest import mock
import uuid
from oslo_serialization import jsonutils
from heat.common import exception as exc
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_handle_update_with_update_only(self):
    self._create_stack(self.template_update_only)
    rsrc = self.stack['deployment_mysql']
    prop_diff = {'input_values': {'foo': 'different'}}
    props = copy.copy(rsrc.properties.data)
    props.update(prop_diff)
    snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    self.deployment.handle_update(json_snippet=snippet, tmpl_diff=None, prop_diff=prop_diff)
    self.rpc_client.show_software_deployment.assert_not_called()