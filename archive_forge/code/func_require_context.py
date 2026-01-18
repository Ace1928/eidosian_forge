import numpy as np
from collections import namedtuple
def require_context(func):
    """
    In the simulator, a context is always "available", so this is a no-op.
    """
    return func