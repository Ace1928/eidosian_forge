import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def custom_help(name, help):
    """Clone a common option overriding the help."""
    import copy
    o = copy.copy(Option.OPTIONS[name])
    o.help = help
    return o