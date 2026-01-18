from __future__ import annotations
from typing import Any
from streamlit import util
from streamlit.runtime.scriptrunner import get_script_run_ctx
def get_locked_cursor(self, **props) -> LockedCursor:
    self._props = props
    return self