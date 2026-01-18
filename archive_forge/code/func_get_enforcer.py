from keystone.common.rbac_enforcer import enforcer
from keystone import conf
def get_enforcer():
    """Entrypoint that must return the raw oslo.policy enforcer obj.

    This is utilized by the command-line policy tools.

    :returns: :class:`oslo_policy.policy.Enforcer`
    """
    CONF(project='keystone')
    return _ENFORCER._enforcer