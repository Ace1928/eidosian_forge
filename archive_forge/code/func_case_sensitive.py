import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
@property
def case_sensitive(self):
    """
        (Read-only)
        ``True`` if path names should be matched sensitive to case; ``False``
        otherwise.
        """
    return self._case_sensitive