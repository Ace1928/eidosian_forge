import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
@property
def patterns(self):
    """
        (Read-only)
        Patterns to allow matching event paths.
        """
    return self._patterns