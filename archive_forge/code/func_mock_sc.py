from unittest import mock
import swiftclient.client
import testscenarios
import testtools
from testtools import matchers
import time
from heatclient.common import deployment_utils
from heatclient import exc
from heatclient.v1 import software_configs
def mock_sc(group=None, config=None, options=None, inputs=None, outputs=None):
    return software_configs.SoftwareConfig(None, {'group': group, 'config': config, 'options': options or {}, 'inputs': inputs or [], 'outputs': outputs or []}, True)