import os
import sys
from breezy import branch, osutils, registry, tests
def generic_help(reg, key):
    help_calls.append(key)
    return 'generic help for {}'.format(key)