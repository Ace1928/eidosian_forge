import linecache
from typing import Any, List, Tuple, Optional
def remember_bpython_input(self, source: str) -> str:
    """Remembers a string of source code, and returns
        a fake filename to use to retrieve it later."""
    filename = f'<bpython-input-{len(self.bpython_history)}>'
    self.bpython_history.append((len(source), None, source.splitlines(True), filename))
    return filename