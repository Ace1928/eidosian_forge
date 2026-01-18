import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def get_negation_name(self):
    if self.name.startswith('no-'):
        return self.name[3:]
    else:
        return 'no-' + self.name