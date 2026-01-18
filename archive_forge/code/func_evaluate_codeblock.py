from __future__ import annotations
from .. import environment, mparser, mesonlib
from .baseobjects import (
from .exceptions import (
from .decorators import FeatureNew
from .disabler import Disabler, is_disabled
from .helpers import default_resolve_key, flatten, resolve_second_level_holders, stringifyUserArguments
from .operator import MesonOperator
from ._unholder import _unholder
import os, copy, re, pathlib
import typing as T
import textwrap
def evaluate_codeblock(self, node: mparser.CodeBlockNode, start: int=0, end: T.Optional[int]=None) -> None:
    if node is None:
        return
    if not isinstance(node, mparser.CodeBlockNode):
        e = InvalidCode('Tried to execute a non-codeblock. Possibly a bug in the parser.')
        e.lineno = node.lineno
        e.colno = node.colno
        raise e
    statements = node.lines[start:end]
    i = 0
    while i < len(statements):
        cur = statements[i]
        try:
            self.current_lineno = cur.lineno
            self.evaluate_statement(cur)
        except Exception as e:
            if getattr(e, 'lineno', None) is None:
                e.lineno = self.current_node.lineno
                e.colno = self.current_node.colno
                e.file = os.path.join(self.source_root, self.subdir, environment.build_filename)
            raise e
        i += 1