from oslo_utils import strutils
from heat.common.i18n import _
def delim_string_to_list(value):
    if value is None:
        return None
    if value == '':
        return []
    return value.split(',')