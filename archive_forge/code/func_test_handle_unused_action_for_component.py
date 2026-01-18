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
def test_handle_unused_action_for_component(self):
    self._create_stack(self.template)
    config = {'id': '48e8ade1-9196-42d5-89a2-f709fde42632', 'group': 'component', 'name': 'myconfig', 'config': {'configs': [{'actions': ['CREATE'], 'config': 'the config', 'tool': 'a_tool'}]}, 'options': {}, 'inputs': [{'name': 'foo', 'type': 'String', 'default': 'baa'}, {'name': 'bar', 'type': 'String', 'default': 'baz'}], 'outputs': []}

    def show_sw_config(*args):
        return config.copy()
    self.rpc_client.show_software_config.side_effect = show_sw_config
    mock_sd = self.mock_deployment()
    self.rpc_client.show_software_deployment.return_value = mock_sd
    self.assertIsNotNone(self.deployment.handle_create())
    self.assertIsNone(self.deployment.handle_delete())