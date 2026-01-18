import os
import sys
from breezy import branch, osutils, registry, tests
def register_stuff(self, a_registry):
    a_registry.register('one', 1)
    a_registry.register('two', 2)
    a_registry.register('four', 4)
    a_registry.register('five', 5)