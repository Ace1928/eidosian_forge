import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def is_hidden(self, name):
    if name == self.name:
        return Option.is_hidden(self, name)
    return getattr(self.registry.get_info(name), 'hidden', False)