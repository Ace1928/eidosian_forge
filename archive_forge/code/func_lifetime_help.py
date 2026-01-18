from neutronclient._i18n import _
from neutronclient.common import exceptions
def lifetime_help(policy):
    lifetime = _("%s lifetime attributes. 'units'-seconds, default:seconds. 'value'-non negative integer, default:3600.") % policy
    return lifetime