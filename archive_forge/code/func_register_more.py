import os
import sys
from breezy import branch, osutils, registry, tests
def register_more():
    _registry.register('hidden', None)