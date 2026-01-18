import ast
import sys
import warnings
from typing import Iterable, Iterator, List, Set, Tuple
from black.mode import VERSION_TO_FEATURES, Feature, TargetVersion, supports_feature
from black.nodes import syms
from blib2to3 import pygram
from blib2to3.pgen2 import driver
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.parse import ParseError
from blib2to3.pgen2.tokenize import TokenError
from blib2to3.pytree import Leaf, Node
def lib2to3_parse(src_txt: str, target_versions: Iterable[TargetVersion]=()) -> Node:
    """Given a string with source, return the lib2to3 Node."""
    if not src_txt.endswith('\n'):
        src_txt += '\n'
    grammars = get_grammars(set(target_versions))
    errors = {}
    for grammar in grammars:
        drv = driver.Driver(grammar)
        try:
            result = drv.parse_string(src_txt, True)
            break
        except ParseError as pe:
            lineno, column = pe.context[1]
            lines = src_txt.splitlines()
            try:
                faulty_line = lines[lineno - 1]
            except IndexError:
                faulty_line = '<line number missing in source>'
            errors[grammar.version] = InvalidInput(f'Cannot parse: {lineno}:{column}: {faulty_line}')
        except TokenError as te:
            lineno, column = te.args[1]
            errors[grammar.version] = InvalidInput(f'Cannot parse: {lineno}:{column}: {te.args[0]}')
    else:
        assert len(errors) >= 1
        exc = errors[max(errors)]
        raise exc from None
    if isinstance(result, Leaf):
        result = Node(syms.file_input, [result])
    return result