from __future__ import annotations
from typing import Any
from streamlit import util
from streamlit.runtime.scriptrunner import get_script_run_ctx
@property
def root_container(self) -> int:
    return self._root_container