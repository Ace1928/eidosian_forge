import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def getblocks(object, lstrip=False, enclosing=False, locate=False):
    """Return a list of source lines and starting line number for an object.
    Interactively-defined objects refer to lines in the interpreter's history.

    If enclosing=True, then also return any enclosing code.
    If lstrip=True, ensure there is no indentation in the first line of code.
    If locate=True, then also return the line number for the block of code.

    DEPRECATED: use 'getsourcelines' instead
    """
    lines, lnum = findsource(object)
    if ismodule(object):
        if lstrip:
            lines = _outdent(lines)
        return ([lines], [0]) if locate is True else [lines]
    indent = indentsize(lines[lnum])
    block = getblock(lines[lnum:])
    if not enclosing or not indent:
        if lstrip:
            block = _outdent(block)
        return ([block], [lnum]) if locate is True else [block]
    pat1 = '^(\\s*def\\s)|(.*(?<!\\w)lambda(:|\\s))'
    pat1 = re.compile(pat1)
    pat2 = '^(\\s*@)'
    pat2 = re.compile(pat2)
    skip = 0
    line = 0
    blocks = []
    _lnum = []
    target = ''.join(block)
    while line <= lnum:
        if pat1.match(lines[line]):
            if not skip:
                try:
                    code = getblock(lines[line:])
                except TokenError:
                    code = [lines[line]]
            if indentsize(lines[line]) > indent:
                line += len(code) - skip
            elif target in ''.join(code):
                blocks.append(code)
                _lnum.append(line - skip)
                line += len(code) - skip
            else:
                line += 1
            skip = 0
        elif pat2.match(lines[line]):
            try:
                code = getblock(lines[line:])
            except TokenError:
                code = [lines[line]]
            skip = 1
            for _line in code[1:]:
                if not pat2.match(_line):
                    break
                skip += 1
            line += skip
        else:
            line += 1
            skip = 0
    if not blocks:
        blocks = [block]
        _lnum = [lnum]
    if lstrip:
        blocks = [_outdent(block) for block in blocks]
    return (blocks, _lnum) if locate is True else blocks