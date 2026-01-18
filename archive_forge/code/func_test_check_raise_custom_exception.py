import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
def test_check_raise_custom_exception(self):
    self.enforcer.set_rules(dict(default=_checks.FalseCheck()))
    creds = {}
    exc = self.assertRaises(MyException, self.enforcer.enforce, 'rule', 'target', creds, True, MyException, 'arg1', 'arg2', kw1='kwarg1', kw2='kwarg2')
    self.assertEqual(('arg1', 'arg2'), exc.args)
    self.assertEqual(dict(kw1='kwarg1', kw2='kwarg2'), exc.kwargs)