import sys
import re
import copy
import time
import os.path
def group_lines(self, input):
    lex = self.lexer.clone()
    lines = [x.rstrip() for x in input.splitlines()]
    for i in xrange(len(lines)):
        j = i + 1
        while lines[i].endswith('\\') and j < len(lines):
            lines[i] = lines[i][:-1] + lines[j]
            lines[j] = ''
            j += 1
    input = '\n'.join(lines)
    lex.input(input)
    lex.lineno = 1
    current_line = []
    while True:
        tok = lex.token()
        if not tok:
            break
        current_line.append(tok)
        if tok.type in self.t_WS and '\n' in tok.value:
            yield current_line
            current_line = []
    if current_line:
        yield current_line