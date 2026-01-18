from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def set_policy_rules(self, rules):
    self.policy.set_rules(oslo_policy.policy.Rules.from_dict(rules), overwrite=True)