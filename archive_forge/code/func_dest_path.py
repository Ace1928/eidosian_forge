import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
@property
def dest_path(self):
    """The destination path of the move event."""
    return self._dest_path