import types
from contextlib import contextmanager
from torch.backends import (
def flags_frozen():
    return not __allow_nonbracketed_mutation_flag