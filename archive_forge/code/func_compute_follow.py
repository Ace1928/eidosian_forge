import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def compute_follow(self, start=None):
    if self.Follow:
        return self.Follow
    if not self.First:
        self.compute_first()
    for k in self.Nonterminals:
        self.Follow[k] = []
    if not start:
        start = self.Productions[1].name
    self.Follow[start] = ['$end']
    while True:
        didadd = False
        for p in self.Productions[1:]:
            for i, B in enumerate(p.prod):
                if B in self.Nonterminals:
                    fst = self._first(p.prod[i + 1:])
                    hasempty = False
                    for f in fst:
                        if f != '<empty>' and f not in self.Follow[B]:
                            self.Follow[B].append(f)
                            didadd = True
                        if f == '<empty>':
                            hasempty = True
                    if hasempty or i == len(p.prod) - 1:
                        for f in self.Follow[p.name]:
                            if f not in self.Follow[B]:
                                self.Follow[B].append(f)
                                didadd = True
        if not didadd:
            break
    return self.Follow