import os
from io import open
import types
from functools import wraps, partial
from contextlib import contextmanager
import sys, re
import sre_parse
import sre_constants
from inspect import getmembers, getmro
from functools import partial, wraps
from itertools import repeat, product
def get_frontend(parser, lexer):
    if parser == 'lalr':
        if lexer is None:
            raise ValueError('The LALR parser requires use of a lexer')
        elif lexer == 'standard':
            return LALR_TraditionalLexer
        elif lexer == 'contextual':
            return LALR_ContextualLexer
        elif issubclass(lexer, Lexer):
            return partial(LALR_CustomLexer, lexer)
        else:
            raise ValueError('Unknown lexer: %s' % lexer)
    elif parser == 'earley':
        if lexer == 'standard':
            return Earley
        elif lexer == 'dynamic':
            return XEarley
        elif lexer == 'dynamic_complete':
            return XEarley_CompleteLex
        elif lexer == 'contextual':
            raise ValueError('The Earley parser does not support the contextual parser')
        else:
            raise ValueError('Unknown lexer: %s' % lexer)
    elif parser == 'cyk':
        if lexer == 'standard':
            return CYK
        else:
            raise ValueError('CYK parser requires using standard parser.')
    else:
        raise ValueError('Unknown parser: %s' % parser)