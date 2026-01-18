import collections
import gc
import time
from tensorflow.python.eager import context
def memory_profiler_is_available():
    return memory_profiler is not None