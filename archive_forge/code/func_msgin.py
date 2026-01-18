import dis
import importlib._bootstrap_external
import importlib.machinery
import marshal
import os
import io
import sys
def msgin(self, *args):
    level = args[0]
    if level <= self.debug:
        self.indent = self.indent + 1
        self.msg(*args)