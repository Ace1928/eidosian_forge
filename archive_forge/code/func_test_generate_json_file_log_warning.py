import operator
from unittest import mock
import warnings
from oslo_config import cfg
import stevedore
import testtools
import yaml
from oslo_policy import generator
from oslo_policy import policy
from oslo_policy.tests import base
from oslo_serialization import jsonutils
@mock.patch.object(generator, 'LOG')
def test_generate_json_file_log_warning(self, mock_log):
    extensions = []
    for name, opts in OPTS.items():
        ext = stevedore.extension.Extension(name=name, entry_point=None, plugin=None, obj=opts)
        extensions.append(ext)
    test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=extensions, namespace=['base_rules', 'rules'])
    output_file = self.get_config_file_fullname('policy.json')
    with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr):
        generator._generate_sample(['base_rules', 'rules'], output_file, output_format='json')
        mock_log.warning.assert_any_call(policy.WARN_JSON)