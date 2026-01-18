import sys
import re
import copy
import time
import os.path
def collect_args(self, tokenlist):
    args = []
    positions = []
    current_arg = []
    nesting = 1
    tokenlen = len(tokenlist)
    i = 0
    while i < tokenlen and tokenlist[i].type in self.t_WS:
        i += 1
    if i < tokenlen and tokenlist[i].value == '(':
        positions.append(i + 1)
    else:
        self.error(self.source, tokenlist[0].lineno, "Missing '(' in macro arguments")
        return (0, [], [])
    i += 1
    while i < tokenlen:
        t = tokenlist[i]
        if t.value == '(':
            current_arg.append(t)
            nesting += 1
        elif t.value == ')':
            nesting -= 1
            if nesting == 0:
                if current_arg:
                    args.append(self.tokenstrip(current_arg))
                    positions.append(i)
                return (i + 1, args, positions)
            current_arg.append(t)
        elif t.value == ',' and nesting == 1:
            args.append(self.tokenstrip(current_arg))
            positions.append(i + 1)
            current_arg = []
        else:
            current_arg.append(t)
        i += 1
    self.error(self.source, tokenlist[-1].lineno, "Missing ')' in macro arguments")
    return (0, [], [])