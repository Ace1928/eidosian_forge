import collections
from .utils import ExplicitEnum, is_torch_available, logging
def expand_frame(self, line):
    self.frame.append(line)