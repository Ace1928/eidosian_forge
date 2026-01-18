import traceback
from contextlib import contextmanager
from typing import List, Any, Dict
from ._compatibility import compatibility
@compatibility(is_backward_compatible=False)
def has_preserved_node_meta() -> bool:
    return should_preserve_node_meta