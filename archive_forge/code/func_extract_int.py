from oslo_utils import strutils
from heat.common.i18n import _
def extract_int(name, value, allow_zero=True, allow_negative=False):
    if value is None:
        return None
    if not strutils.is_int_like(value):
        raise ValueError(_("Only integer is acceptable by '%(name)s'.") % {'name': name})
    if value in ('0', 0):
        if allow_zero:
            return int(value)
        raise ValueError(_("Only non-zero integer is acceptable by '%(name)s'.") % {'name': name})
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(_("Value '%(value)s' is invalid for '%(name)s' which only accepts integer.") % {'name': name, 'value': value})
    if allow_negative is False and result < 0:
        raise ValueError(_("Value '%(value)s' is invalid for '%(name)s' which only accepts non-negative integer.") % {'name': name, 'value': value})
    return result