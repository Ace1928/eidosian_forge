import re
import sys
import types
import copy
import os
import inspect
def runmain(lexer=None, data=None):
    if not data:
        try:
            filename = sys.argv[1]
            f = open(filename)
            data = f.read()
            f.close()
        except IndexError:
            sys.stdout.write('Reading from standard input (type EOF to end):\n')
            data = sys.stdin.read()
    if lexer:
        _input = lexer.input
    else:
        _input = input
    _input(data)
    if lexer:
        _token = lexer.token
    else:
        _token = token
    while True:
        tok = _token()
        if not tok:
            break
        sys.stdout.write('(%s,%r,%d,%d)\n' % (tok.type, tok.value, tok.lineno, tok.lexpos))