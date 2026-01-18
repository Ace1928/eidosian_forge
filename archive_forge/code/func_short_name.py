import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def short_name(self):
    if self._short_name:
        return self._short_name