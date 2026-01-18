from debtcollector import removals
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
@removals.remove(message='Use %s.check instead.' % deprecation_msg, version='3.9.0', removal_version='4.0.0')
def check_implied(self, prior_role, implied_role, **kwargs):
    return InferenceRuleManager(self.client).check(prior_role, implied_role)