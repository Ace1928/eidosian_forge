import ast
import difflib
from collections.abc import Mapping
from typing import Any, Callable, Dict
def evaluate_if(if_statement, state, tools):
    result = None
    if evaluate_condition(if_statement.test, state, tools):
        for line in if_statement.body:
            line_result = evaluate_ast(line, state, tools)
            if line_result is not None:
                result = line_result
    else:
        for line in if_statement.orelse:
            line_result = evaluate_ast(line, state, tools)
            if line_result is not None:
                result = line_result
    return result