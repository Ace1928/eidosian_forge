from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def char_min(char_dicts, min_prop):
    if char_dicts:
        return sum((char_dict[min_prop] for char_dict in char_dicts))
    return 0