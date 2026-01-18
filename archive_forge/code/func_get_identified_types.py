import collections
from llvmlite.ir import context, values, types, _utils
def get_identified_types(self):
    return self.context.identified_types