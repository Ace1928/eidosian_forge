from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
def descr_arg(i, a):
    if i in self.arg_attributes:
        attrs = ' '.join(self.arg_attributes[i]._to_list(a.type)) + ' '
    else:
        attrs = ''
    return '{0} {1}{2}'.format(a.type, attrs, a.get_reference())