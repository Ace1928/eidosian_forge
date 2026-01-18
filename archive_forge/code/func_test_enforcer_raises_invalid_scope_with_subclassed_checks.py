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
def test_enforcer_raises_invalid_scope_with_subclassed_checks(self):
    self.conf.set_override('enforce_scope', True, group='oslo_policy')
    rule = _checks.TrueCheck()
    rule.scope_types = ['domain']
    ctx = context.RequestContext(system_scope='all', roles=['admin'])
    self.assertRaises(policy.InvalidScope, self.enforcer.enforce, rule, {}, ctx, do_raise=True)
    self.assertFalse(self.enforcer.enforce(rule, {}, ctx, do_raise=False))