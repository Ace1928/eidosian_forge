import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
def declared_identifiers(self):
    return self.body_decl.allargnames