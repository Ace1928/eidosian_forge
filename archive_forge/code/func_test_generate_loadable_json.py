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
def test_generate_loadable_json(self):
    extensions = []
    for name, opts in OPTS.items():
        ext = stevedore.extension.Extension(name=name, entry_point=None, plugin=None, obj=opts)
        extensions.append(ext)
    test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=extensions, namespace=['base_rules', 'rules'])
    output_file = self.get_config_file_fullname('policy.json')
    with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
        generator._generate_sample(['base_rules', 'rules'], output_file, output_format='json', include_help=False)
        mock_ext_mgr.assert_called_once_with('oslo.policy.policies', names=['base_rules', 'rules'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
    self.enforcer.load_rules()
    self.assertIn('owner', self.enforcer.rules)
    self.assertIn('admin', self.enforcer.rules)
    self.assertIn('admin_or_owner', self.enforcer.rules)
    self.assertEqual('project_id:%(project_id)s', str(self.enforcer.rules['owner']))
    self.assertEqual('is_admin:True', str(self.enforcer.rules['admin']))
    self.assertEqual('(rule:admin or rule:owner)', str(self.enforcer.rules['admin_or_owner']))