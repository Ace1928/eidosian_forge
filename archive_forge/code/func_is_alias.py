import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def is_alias(self, name):
    """Check whether a particular format is an alias."""
    if name == self.name:
        return False
    return name in self.registry.aliases()