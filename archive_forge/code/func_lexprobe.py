import sys
import re
import copy
import time
import os.path
def lexprobe(self):
    self.lexer.input('identifier')
    tok = self.lexer.token()
    if not tok or tok.value != 'identifier':
        print("Couldn't determine identifier type")
    else:
        self.t_ID = tok.type
    self.lexer.input('12345')
    tok = self.lexer.token()
    if not tok or int(tok.value) != 12345:
        print("Couldn't determine integer type")
    else:
        self.t_INTEGER = tok.type
        self.t_INTEGER_TYPE = type(tok.value)
    self.lexer.input('"filename"')
    tok = self.lexer.token()
    if not tok or tok.value != '"filename"':
        print("Couldn't determine string type")
    else:
        self.t_STRING = tok.type
    self.lexer.input('  ')
    tok = self.lexer.token()
    if not tok or tok.value != '  ':
        self.t_SPACE = None
    else:
        self.t_SPACE = tok.type
    self.lexer.input('\n')
    tok = self.lexer.token()
    if not tok or tok.value != '\n':
        self.t_NEWLINE = None
        print("Couldn't determine token for newlines")
    else:
        self.t_NEWLINE = tok.type
    self.t_WS = (self.t_SPACE, self.t_NEWLINE)
    chars = ['<', '>', '#', '##', '\\', '(', ')', ',', '.']
    for c in chars:
        self.lexer.input(c)
        tok = self.lexer.token()
        if not tok or tok.value != c:
            print("Unable to lex '%s' required for preprocessor" % c)