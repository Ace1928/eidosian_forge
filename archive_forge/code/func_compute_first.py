import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def compute_first(self):
    if self.First:
        return self.First
    for t in self.Terminals:
        self.First[t] = [t]
    self.First['$end'] = ['$end']
    for n in self.Nonterminals:
        self.First[n] = []
    while True:
        some_change = False
        for n in self.Nonterminals:
            for p in self.Prodnames[n]:
                for f in self._first(p.prod):
                    if f not in self.First[n]:
                        self.First[n].append(f)
                        some_change = True
        if not some_change:
            break
    return self.First