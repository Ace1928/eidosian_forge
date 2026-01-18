import ast
import sys
import dis
from types import CodeType, FrameType
from typing import Any, Callable, Iterator, Optional, Sequence, Set, Tuple, Type, Union, cast
from .executing import EnhancedAST, NotOneValueFound, Source, only, function_node_types, assert_
from ._exceptions import KnownIssue, VerifierFailure
from functools import lru_cache
def known_issues(self, node: EnhancedAST, instruction: dis.Instruction) -> None:
    if instruction.opname in ('COMPARE_OP', 'IS_OP', 'CONTAINS_OP') and isinstance(node, types_cmp_issue):
        if isinstance(node, types_cmp_issue_fix):
            comparisons = [n for n in ast.walk(node.test) if isinstance(n, ast.Compare) and len(n.ops) > 1]
            assert_(comparisons, 'expected at least one comparison')
            if len(comparisons) == 1:
                node = self.result = cast(EnhancedAST, comparisons[0])
            else:
                raise KnownIssue('multiple chain comparison inside %s can not be fixed' % node)
        else:
            raise KnownIssue('chain comparison inside %s can not be fixed' % node)
    if sys.version_info[:3] == (3, 11, 1) and isinstance(node, ast.Compare) and (instruction.opname == 'CALL') and any((isinstance(n, ast.Assert) for n in node_and_parents(node))):
        raise KnownIssue('known bug in 3.11.1 https://github.com/python/cpython/issues/95921')
    if isinstance(node, ast.Assert):
        raise KnownIssue('assert')
    if any((isinstance(n, ast.pattern) for n in node_and_parents(node))):
        raise KnownIssue('pattern matching ranges seems to be wrong')
    if sys.version_info >= (3, 12) and isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == 'super'):
        func = node.parent
        while hasattr(func, 'parent') and (not isinstance(func, (ast.AsyncFunctionDef, ast.FunctionDef))):
            func = func.parent
        first_arg = None
        if hasattr(func, 'args'):
            args = [*func.args.posonlyargs, *func.args.args]
            if args:
                first_arg = args[0].arg
        if (instruction.opname, instruction.argval) in [('LOAD_DEREF', '__class__'), ('LOAD_FAST', first_arg), ('LOAD_DEREF', first_arg)]:
            raise KnownIssue('super optimization')
    if self.is_except_cleanup(instruction, node):
        raise KnownIssue('exeption cleanup does not belong to the last node in a except block')
    if instruction.opname == 'STORE_NAME' and instruction.argval == '__classcell__':
        raise KnownIssue('store __classcell__')
    if instruction.opname == 'CALL' and (not isinstance(node, ast.Call)) and any((isinstance(p, ast.Assert) for p in parents(node))) and (sys.version_info >= (3, 11, 2)):
        raise KnownIssue('exception generation maps to condition')