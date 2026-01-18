import linecache
from typing import Any, List, Tuple, Optional
def filename_for_console_input(code_string: str) -> str:
    """Remembers a string of source code, and returns
    a fake filename to use to retrieve it later."""
    if isinstance(linecache.cache, BPythonLinecache):
        return linecache.cache.remember_bpython_input(code_string)
    else:
        return '<input>'