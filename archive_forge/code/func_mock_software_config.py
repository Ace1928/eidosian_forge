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
def mock_software_config(self):
    config = {'group': 'Test::Group', 'name': 'myconfig', 'config': 'the config', 'options': {}, 'inputs': [{'name': 'foo', 'type': 'String', 'default': 'baa'}, {'name': 'bar', 'type': 'String', 'default': 'baz'}, {'name': 'trigger_replace', 'type': 'String', 'default': 'default_value', 'replace_on_change': True}], 'outputs': []}
    derived_config = copy.deepcopy(config)
    values = {'foo': 'bar'}
    inputs = derived_config['inputs']
    for i in inputs:
        i['value'] = values.get(i['name'], i['default'])
    inputs.append({'name': 'deploy_signal_transport', 'type': 'String', 'value': 'NO_SIGNAL'})
    configs = {'0ff2e903-78d7-4cca-829e-233af3dae705': config, '48e8ade1-9196-42d5-89a2-f709fde42632': config, '9966c8e7-bc9c-42de-aa7d-f2447a952cb2': derived_config}

    def copy_config(context, config_id):
        config = configs[config_id].copy()
        config['id'] = config_id
        return config
    self.rpc_client.show_software_config.side_effect = copy_config
    return config