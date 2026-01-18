import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def not_allowed(self, node):
    self.error(node, '%s statements are not allowed.' % node.__class__.__name__)