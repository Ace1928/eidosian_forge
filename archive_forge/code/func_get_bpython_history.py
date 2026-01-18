import linecache
from typing import Any, List, Tuple, Optional
def get_bpython_history(self, key: str) -> Tuple[int, None, List[str], str]:
    """Given a filename provided by remember_bpython_input,
        returns the associated source string."""
    try:
        idx = int(key.split('-')[2][:-1])
        return self.bpython_history[idx]
    except (IndexError, ValueError):
        raise KeyError