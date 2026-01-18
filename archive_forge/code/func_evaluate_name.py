import ast
import difflib
from collections.abc import Mapping
from typing import Any, Callable, Dict
def evaluate_name(name, state, tools):
    if name.id in state:
        return state[name.id]
    close_matches = difflib.get_close_matches(name.id, list(state.keys()))
    if len(close_matches) > 0:
        return state[close_matches[0]]
    raise InterpretorError(f'The variable `{name.id}` is not defined.')