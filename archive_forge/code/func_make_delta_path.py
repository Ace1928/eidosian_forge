from __future__ import annotations
from typing import Any
from streamlit import util
from streamlit.runtime.scriptrunner import get_script_run_ctx
def make_delta_path(root_container: int, parent_path: tuple[int, ...], index: int) -> list[int]:
    delta_path = [root_container]
    delta_path.extend(parent_path)
    delta_path.append(index)
    return delta_path