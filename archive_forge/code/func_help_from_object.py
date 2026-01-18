import os
import sys
from breezy import branch, osutils, registry, tests
def help_from_object(reg, key):
    obj = reg.get(key)
    return obj.help()