from llvmlite.ir import _utils
from llvmlite.ir import types
def get_identified_type(self, name):
    if name not in self.identified_types:
        self.scope.register(name)
        ty = types.IdentifiedStructType(self, name)
        self.identified_types[name] = ty
    else:
        ty = self.identified_types[name]
    return ty