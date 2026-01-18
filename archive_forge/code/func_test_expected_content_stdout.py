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
def test_expected_content_stdout(self):
    extensions = []
    for name, opts in OPTS.items():
        ext = stevedore.extension.Extension(name=name, entry_point=None, plugin=None, obj=opts)
        extensions.append(ext)
    test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=extensions, namespace=['base_rules', 'rules'])
    expected = '{\n    "admin": "is_admin:True",\n    "owner": "project_id:%(project_id)s",\n    "shared": "field:networks:shared=True",\n    "admin_or_owner": "rule:admin or rule:owner"\n}\n'
    stdout = self._capture_stdout()
    with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
        generator._generate_sample(['base_rules', 'rules'], output_file=None, output_format='json')
        mock_ext_mgr.assert_called_once_with('oslo.policy.policies', names=['base_rules', 'rules'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
    self.assertEqual(expected, stdout.getvalue())