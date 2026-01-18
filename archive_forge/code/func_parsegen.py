import sys
import re
import copy
import time
import os.path
def parsegen(self, input, source=None):
    t = trigraph(input)
    lines = self.group_lines(t)
    if not source:
        source = ''
    self.define('__FILE__ "%s"' % source)
    self.source = source
    chunk = []
    enable = True
    iftrigger = False
    ifstack = []
    for x in lines:
        for i, tok in enumerate(x):
            if tok.type not in self.t_WS:
                break
        if tok.value == '#':
            for tok in x:
                if tok.type in self.t_WS and '\n' in tok.value:
                    chunk.append(tok)
            dirtokens = self.tokenstrip(x[i + 1:])
            if dirtokens:
                name = dirtokens[0].value
                args = self.tokenstrip(dirtokens[1:])
            else:
                name = ''
                args = []
            if name == 'define':
                if enable:
                    for tok in self.expand_macros(chunk):
                        yield tok
                    chunk = []
                    self.define(args)
            elif name == 'include':
                if enable:
                    for tok in self.expand_macros(chunk):
                        yield tok
                    chunk = []
                    oldfile = self.macros['__FILE__']
                    for tok in self.include(args):
                        yield tok
                    self.macros['__FILE__'] = oldfile
                    self.source = source
            elif name == 'undef':
                if enable:
                    for tok in self.expand_macros(chunk):
                        yield tok
                    chunk = []
                    self.undef(args)
            elif name == 'ifdef':
                ifstack.append((enable, iftrigger))
                if enable:
                    if not args[0].value in self.macros:
                        enable = False
                        iftrigger = False
                    else:
                        iftrigger = True
            elif name == 'ifndef':
                ifstack.append((enable, iftrigger))
                if enable:
                    if args[0].value in self.macros:
                        enable = False
                        iftrigger = False
                    else:
                        iftrigger = True
            elif name == 'if':
                ifstack.append((enable, iftrigger))
                if enable:
                    result = self.evalexpr(args)
                    if not result:
                        enable = False
                        iftrigger = False
                    else:
                        iftrigger = True
            elif name == 'elif':
                if ifstack:
                    if ifstack[-1][0]:
                        if enable:
                            enable = False
                        elif not iftrigger:
                            result = self.evalexpr(args)
                            if result:
                                enable = True
                                iftrigger = True
                else:
                    self.error(self.source, dirtokens[0].lineno, 'Misplaced #elif')
            elif name == 'else':
                if ifstack:
                    if ifstack[-1][0]:
                        if enable:
                            enable = False
                        elif not iftrigger:
                            enable = True
                            iftrigger = True
                else:
                    self.error(self.source, dirtokens[0].lineno, 'Misplaced #else')
            elif name == 'endif':
                if ifstack:
                    enable, iftrigger = ifstack.pop()
                else:
                    self.error(self.source, dirtokens[0].lineno, 'Misplaced #endif')
            else:
                pass
        elif enable:
            chunk.extend(x)
    for tok in self.expand_macros(chunk):
        yield tok
    chunk = []