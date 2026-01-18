from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def parse_pspec(pargs, flags):
    """
  Parse a positional argument specification.
  """
    out = []
    if pargs is None:
        pargs = ZERO_OR_MORE
    if isinstance(pargs, STRING_TYPES + (int,)):
        return [PositionalSpec(pargs, flags=flags, legacy=True)]
    if flags:
        raise UserError("Illegal use of top-level 'flags' keyword with new-style positional argument declaration")
    if isinstance(pargs, dict):
        pargs = [pargs]
    for pargdecl in pargs:
        if isinstance(pargdecl, STRING_TYPES + (int,)):
            out.append(PositionalSpec(pargdecl))
            continue
        if isinstance(pargdecl, dict):
            if 'npargs' not in pargdecl:
                pargdecl = dict(pargdecl)
                pargdecl['nargs'] = ZERO_OR_MORE
            out.append(PositionalSpec(**pargdecl))
            continue
        if isinstance(pargdecl, (list, tuple)):
            args = list(pargdecl)
            kwargs = {}
            if isinstance(args[-1], dict):
                kwargs = args.pop(-1)
            out.append(PositionalSpec(*args, **kwargs))
    return out