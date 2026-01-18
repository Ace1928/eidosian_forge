from oslo_utils import strutils
from heat.common.i18n import _
def extract_bool(name, value):
    """Convert any true/false string to its corresponding boolean value.

    Value is case insensitive.
    """
    if str(value).lower() not in ('true', 'false'):
        raise ValueError(_('Unrecognized value "%(value)s" for "%(name)s", acceptable values are: true, false.') % {'value': value, 'name': name})
    return strutils.bool_from_string(value, strict=True)