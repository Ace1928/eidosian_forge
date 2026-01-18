import collections
import copy
from heat.common.i18n import _
from heat.common import exception
from heat.engine import constraints
from heat.engine import parameters
from heat.engine import properties
def check_io_schema_list(io_configs):
    """Check that an input or output schema list is of the correct type.

    Raises TypeError if the list itself is not a list, or if any of the
    members are not dicts.
    """
    if not isinstance(io_configs, collections.abc.Sequence) or isinstance(io_configs, collections.abc.Mapping) or isinstance(io_configs, str):
        raise TypeError('Software Config I/O Schema must be in a list')
    if not all((isinstance(conf, collections.abc.Mapping) for conf in io_configs)):
        raise TypeError('Software Config I/O Schema must be a dict')