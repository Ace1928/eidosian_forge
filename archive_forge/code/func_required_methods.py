from oslo_log import log
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.auth import core
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.identity.backends import resource_options as ro
@property
def required_methods(self):
    if not self.__required_methods:
        mfa_rules = self.user['options'].get(ro.MFA_RULES_OPT.option_name, [])
        rules = core.UserMFARulesValidator._parse_rule_structure(mfa_rules, self.user_id)
        methods = set(self.methods)
        active_methods = set(core.AUTH_METHODS.keys())
        required_auth_methods = []
        for r in rules:
            r_set = set(r).intersection(active_methods)
            if r_set.intersection(methods):
                required_auth_methods.append(list(r_set))
        self.__required_methods = required_auth_methods
    return self.__required_methods