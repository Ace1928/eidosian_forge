import re
from mako import exceptions
from mako import pyparser
@property
def allargnames(self):
    return tuple(self.argnames) + tuple(self.kwargnames)