import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
def camel_case(self):
    return ''.join(self.capitalized())