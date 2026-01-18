import ast
import sys
import dis
from types import CodeType, FrameType
from typing import Any, Callable, Iterator, Optional, Sequence, Set, Tuple, Type, Union, cast
from .executing import EnhancedAST, NotOneValueFound, Source, only, function_node_types, assert_
from ._exceptions import KnownIssue, VerifierFailure
from functools import lru_cache
def test_for_decorator(self, node: EnhancedAST, index: int) -> None:
    if isinstance(node.parent, (ast.ClassDef, function_node_types)) and node in node.parent.decorator_list:
        node_func = node.parent
        while True:
            if not ((self.opname(index - 4) == 'PRECALL' or sys.version_info >= (3, 12)) and self.opname(index) == 'CALL'):
                break
            index += 2
            while self.opname(index) in ('CACHE', 'EXTENDED_ARG'):
                index += 2
            if self.opname(index).startswith('STORE_') and self.find_node(index) == node_func:
                self.result = node_func
                self.decorator = node
                return
            if sys.version_info < (3, 12):
                index += 4