import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def iter_switches(self):
    """Iterate through the list of switches provided by the option

        :return: an iterator of (name, short_name, argname, help)
        """
    yield from Option.iter_switches(self)
    if self.value_switches:
        for key in sorted(self.registry.keys()):
            yield (key, None, None, self.registry.get_help(key))