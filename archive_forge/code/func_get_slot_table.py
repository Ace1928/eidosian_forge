from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def get_slot_table(compiler_directives):
    if not compiler_directives:
        from .Options import get_directive_defaults
        compiler_directives = get_directive_defaults()
    old_binops = compiler_directives['c_api_binop_methods']
    key = (old_binops,)
    if key not in _slot_table_dict:
        _slot_table_dict[key] = SlotTable(old_binops=old_binops)
    return _slot_table_dict[key]