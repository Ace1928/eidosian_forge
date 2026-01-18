import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def compute_nullable_nonterminals(self):
    nullable = set()
    num_nullable = 0
    while True:
        for p in self.grammar.Productions[1:]:
            if p.len == 0:
                nullable.add(p.name)
                continue
            for t in p.prod:
                if t not in nullable:
                    break
            else:
                nullable.add(p.name)
        if len(nullable) == num_nullable:
            break
        num_nullable = len(nullable)
    return nullable