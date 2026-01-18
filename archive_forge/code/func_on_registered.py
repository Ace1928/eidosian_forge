from typing import (
from .constants import (
from .exceptions import (
from .utils import (
def on_registered(self) -> None:
    """
        Called by cmd2.Cmd after a CommandSet is registered and all its commands have been added to the CLI.
        Subclasses can override this to perform custom steps related to the newly added commands (e.g. setting
        them to a disabled state).
        """
    pass