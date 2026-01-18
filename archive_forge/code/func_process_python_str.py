import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def process_python_str(python_str: str) -> Value:
    value = ast.literal_eval(python_str)
    return Value(str(value))