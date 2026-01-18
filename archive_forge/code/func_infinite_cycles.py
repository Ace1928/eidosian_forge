import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def infinite_cycles(self):
    terminates = {}
    for t in self.Terminals:
        terminates[t] = True
    terminates['$end'] = True
    for n in self.Nonterminals:
        terminates[n] = False
    while True:
        some_change = False
        for n, pl in self.Prodnames.items():
            for p in pl:
                for s in p.prod:
                    if not terminates[s]:
                        p_terminates = False
                        break
                else:
                    p_terminates = True
                if p_terminates:
                    if not terminates[n]:
                        terminates[n] = True
                        some_change = True
                    break
        if not some_change:
            break
    infinite = []
    for s, term in terminates.items():
        if not term:
            if s not in self.Prodnames and s not in self.Terminals and (s != 'error'):
                pass
            else:
                infinite.append(s)
    return infinite