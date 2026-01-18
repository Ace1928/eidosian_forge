import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
@property
def ignore_regexes(self):
    """
        (Read-only)
        Regexes to ignore matching event paths.
        """
    return self._ignore_regexes