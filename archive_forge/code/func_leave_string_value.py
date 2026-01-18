from typing import Any, Collection, Optional
from ..language.ast import Node, OperationType
from .block_string import print_block_string
from .print_string import print_string
from .visitor import visit, Visitor
@staticmethod
def leave_string_value(node: PrintedNode, *_args: Any) -> str:
    if node.block:
        return print_block_string(node.value)
    return print_string(node.value)