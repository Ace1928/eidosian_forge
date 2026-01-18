from io import BytesIO
from typing import Callable, Dict, List, Tuple
from .. import errors, osutils, registry
def relpath(self):
    """Relative path of file to tree-root."""
    return self._relpath