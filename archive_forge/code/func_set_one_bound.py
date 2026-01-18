import re
from . import constants as const
def set_one_bound(bound_type, value):
    variable_dict[var_name][BOUNDS_EQUIV[bound_type]] = value