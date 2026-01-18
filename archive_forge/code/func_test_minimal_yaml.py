from unittest import mock
import yaml
from heat.common import environment_format
from heat.tests import common
def test_minimal_yaml(self):
    yaml1 = ''
    yaml2 = '\nparameters: {}\nencrypted_param_names: []\nparameter_defaults: {}\nevent_sinks: []\nresource_registry: {}\n'
    tpl1 = environment_format.parse(yaml1)
    environment_format.default_for_missing(tpl1)
    tpl2 = environment_format.parse(yaml2)
    self.assertEqual(tpl1, tpl2)