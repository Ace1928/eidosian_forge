import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def add_production(self, prodname, syms, func=None, file='', line=0):
    if prodname in self.Terminals:
        raise GrammarError('%s:%d: Illegal rule name %r. Already defined as a token' % (file, line, prodname))
    if prodname == 'error':
        raise GrammarError('%s:%d: Illegal rule name %r. error is a reserved word' % (file, line, prodname))
    if not _is_identifier.match(prodname):
        raise GrammarError('%s:%d: Illegal rule name %r' % (file, line, prodname))
    for n, s in enumerate(syms):
        if s[0] in '\'"':
            try:
                c = eval(s)
                if len(c) > 1:
                    raise GrammarError('%s:%d: Literal token %s in rule %r may only be a single character' % (file, line, s, prodname))
                if c not in self.Terminals:
                    self.Terminals[c] = []
                syms[n] = c
                continue
            except SyntaxError:
                pass
        if not _is_identifier.match(s) and s != '%prec':
            raise GrammarError('%s:%d: Illegal name %r in rule %r' % (file, line, s, prodname))
    if '%prec' in syms:
        if syms[-1] == '%prec':
            raise GrammarError('%s:%d: Syntax error. Nothing follows %%prec' % (file, line))
        if syms[-2] != '%prec':
            raise GrammarError('%s:%d: Syntax error. %%prec can only appear at the end of a grammar rule' % (file, line))
        precname = syms[-1]
        prodprec = self.Precedence.get(precname)
        if not prodprec:
            raise GrammarError('%s:%d: Nothing known about the precedence of %r' % (file, line, precname))
        else:
            self.UsedPrecedence.add(precname)
        del syms[-2:]
    else:
        precname = rightmost_terminal(syms, self.Terminals)
        prodprec = self.Precedence.get(precname, ('right', 0))
    map = '%s -> %s' % (prodname, syms)
    if map in self.Prodmap:
        m = self.Prodmap[map]
        raise GrammarError('%s:%d: Duplicate rule %s. ' % (file, line, m) + 'Previous definition at %s:%d' % (m.file, m.line))
    pnumber = len(self.Productions)
    if prodname not in self.Nonterminals:
        self.Nonterminals[prodname] = []
    for t in syms:
        if t in self.Terminals:
            self.Terminals[t].append(pnumber)
        else:
            if t not in self.Nonterminals:
                self.Nonterminals[t] = []
            self.Nonterminals[t].append(pnumber)
    p = Production(pnumber, prodname, syms, prodprec, func, file, line)
    self.Productions.append(p)
    self.Prodmap[map] = p
    try:
        self.Prodnames[prodname].append(p)
    except KeyError:
        self.Prodnames[prodname] = [p]