from __future__ import annotations
import ast
import json
import os
from io import StringIO
from sys import version_info
from typing import IO, TYPE_CHECKING, Any, Callable, List, Optional, Type, Union
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain_community.tools import BaseTool, Tool
from langchain_community.tools.e2b_data_analysis.unparse import Unparser
def add_last_line_print(code: str) -> str:
    """Add print statement to the last line if it's missing.

    Sometimes, the LLM-generated code doesn't have `print(variable_name)`, instead the
        LLM tries to print the variable only by writing `variable_name` (as you would in
        REPL, for example).

    This methods checks the AST of the generated Python code and adds the print
        statement to the last line if it's missing.
    """
    tree = ast.parse(code)
    node = tree.body[-1]
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        if isinstance(node.value.func, ast.Name) and node.value.func.id == 'print':
            return _unparse(tree)
    if isinstance(node, ast.Expr):
        tree.body[-1] = ast.Expr(value=ast.Call(func=ast.Name(id='print', ctx=ast.Load()), args=[node.value], keywords=[]))
    return _unparse(tree)