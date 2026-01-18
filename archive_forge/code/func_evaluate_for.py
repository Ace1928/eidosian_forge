import ast
import difflib
from collections.abc import Mapping
from typing import Any, Callable, Dict
def evaluate_for(for_loop, state, tools):
    result = None
    iterator = evaluate_ast(for_loop.iter, state, tools)
    for counter in iterator:
        state[for_loop.target.id] = counter
        for expression in for_loop.body:
            line_result = evaluate_ast(expression, state, tools)
            if line_result is not None:
                result = line_result
    return result