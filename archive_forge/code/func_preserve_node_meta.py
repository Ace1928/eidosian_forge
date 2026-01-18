import traceback
from contextlib import contextmanager
from typing import List, Any, Dict
from ._compatibility import compatibility
@compatibility(is_backward_compatible=False)
@contextmanager
def preserve_node_meta():
    global should_preserve_node_meta
    saved_should_preserve_node_meta = should_preserve_node_meta
    try:
        should_preserve_node_meta = True
        yield
    finally:
        should_preserve_node_meta = saved_should_preserve_node_meta