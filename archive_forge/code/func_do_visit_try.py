import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def do_visit_try(self, node):
    self.fill('try')
    with self.block():
        self.traverse(node.body)
    for ex in node.handlers:
        self.traverse(ex)
    if node.orelse:
        self.fill('else')
        with self.block():
            self.traverse(node.orelse)
    if node.finalbody:
        self.fill('finally')
        with self.block():
            self.traverse(node.finalbody)