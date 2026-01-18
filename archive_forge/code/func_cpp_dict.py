from torch import nn
@property
def cpp_dict(self):
    return getattr(self.cpp_module, self.attr)