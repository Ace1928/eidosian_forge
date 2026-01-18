import unittest
from traits.api import (
@on_trait_change('int1, int2,')
def arg_check(self, object, name, old, new):
    pass