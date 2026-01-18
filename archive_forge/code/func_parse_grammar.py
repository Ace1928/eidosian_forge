import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def parse_grammar(doc, file, line):
    grammar = []
    pstrings = doc.splitlines()
    lastp = None
    dline = line
    for ps in pstrings:
        dline += 1
        p = ps.split()
        if not p:
            continue
        try:
            if p[0] == '|':
                if not lastp:
                    raise SyntaxError("%s:%d: Misplaced '|'" % (file, dline))
                prodname = lastp
                syms = p[1:]
            else:
                prodname = p[0]
                lastp = prodname
                syms = p[2:]
                assign = p[1]
                if assign != ':' and assign != '::=':
                    raise SyntaxError("%s:%d: Syntax error. Expected ':'" % (file, dline))
            grammar.append((file, dline, prodname, syms))
        except SyntaxError:
            raise
        except Exception:
            raise SyntaxError('%s:%d: Syntax error in rule %r' % (file, dline, ps.strip()))
    return grammar