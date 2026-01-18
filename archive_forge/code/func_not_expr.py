import ast
import dataclasses
import enum
import re
import types
from typing import Callable
from typing import Iterator
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
def not_expr(s: Scanner) -> ast.expr:
    if s.accept(TokenType.NOT):
        return ast.UnaryOp(ast.Not(), not_expr(s))
    if s.accept(TokenType.LPAREN):
        ret = expr(s)
        s.accept(TokenType.RPAREN, reject=True)
        return ret
    ident = s.accept(TokenType.IDENT)
    if ident:
        return ast.Name(IDENT_PREFIX + ident.value, ast.Load())
    s.reject((TokenType.NOT, TokenType.LPAREN, TokenType.IDENT))