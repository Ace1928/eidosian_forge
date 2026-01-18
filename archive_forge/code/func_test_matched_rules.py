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
@mock.patch('warnings.warn')
def test_matched_rules(self, mock_warn):
    extensions = []
    for name, opts in OPTS.items():
        ext = stevedore.extension.Extension(name=name, entry_point=None, plugin=None, obj=opts)
        extensions.append(ext)
    test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=extensions, namespace=['base_rules', 'rules'])
    sample_file = self.get_config_file_fullname('policy-sample.yaml')
    with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr):
        generator._generate_sample(['base_rules', 'rules'], sample_file, include_help=False)
    enforcer = policy.Enforcer(self.conf, policy_file='policy-sample.yaml')
    enforcer.register_default(policy.RuleDefault('admin', 'is_admin:True'))
    enforcer.register_default(policy.RuleDefault('owner', 'project_id:%(project_id)s'))
    deprecated_rule = policy.DeprecatedRule(name='old_foo', check_str='role:bar', deprecated_reason='reason', deprecated_since='T')
    enforcer.register_default(policy.RuleDefault(name='foo', check_str='role:foo', deprecated_rule=deprecated_rule))
    ext = stevedore.extension.Extension(name='testing', entry_point=None, plugin=None, obj=enforcer)
    test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=[ext], namespace='testing')
    stdout = self._capture_stdout()
    with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
        generator._list_redundant(namespace='testing')
        mock_ext_mgr.assert_called_once_with('oslo.policy.enforcer', names=['testing'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
    matches = [line.split(': ', 1) for line in stdout.getvalue().splitlines()]
    matches.sort(key=operator.itemgetter(0))
    opt0 = matches[0]
    self.assertEqual('"admin"', opt0[0])
    self.assertEqual('"is_admin:True"', opt0[1])
    opt1 = matches[1]
    self.assertEqual('"owner"', opt1[0])
    self.assertEqual('"project_id:%(project_id)s"', opt1[1])
    self.assertFalse(mock_warn.called, 'Deprecation warnings not suppressed.')