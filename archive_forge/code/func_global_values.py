import collections
from llvmlite.ir import context, values, types, _utils
@property
def global_values(self):
    """
        An iterable of global values in this module.
        """
    return self.globals.values()