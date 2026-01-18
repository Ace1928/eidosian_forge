import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
def capitalized(self):
    return WordSet((word.capitalize() for word in self))