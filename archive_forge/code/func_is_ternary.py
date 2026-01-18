import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
def is_ternary(self, keyword):
    """return true if the given keyword is a ternary keyword
        for this ControlLine"""
    cases = {'if': {'else', 'elif'}, 'try': {'except', 'finally'}, 'for': {'else'}}
    return keyword in cases.get(self.keyword, set())