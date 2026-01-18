from keystone.common import resource_options
from keystone.common.validation import parameter_types
from keystone.i18n import _
def register_user_options():
    for opt in [IGNORE_CHANGE_PASSWORD_OPT, IGNORE_PASSWORD_EXPIRY_OPT, IGNORE_LOCKOUT_ATTEMPT_OPT, LOCK_PASSWORD_OPT, IGNORE_USER_INACTIVITY_OPT, MFA_RULES_OPT, MFA_ENABLED_OPT]:
        USER_OPTIONS_REGISTRY.register_option(opt)