import linecache
from typing import Any, List, Tuple, Optional
def is_bpython_filename(self, fname: Any) -> bool:
    return isinstance(fname, str) and fname.startswith('<bpython-input-')