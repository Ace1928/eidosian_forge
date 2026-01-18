from itertools import repeat
from autograd.wrap_util import wraps
from autograd.util import subvals, toposort
from autograd.tracer import trace, Node
from functools import partial
def initialize_root(self):
    self.value = None
    self.recipe = (lambda x: x, (), {}, [])