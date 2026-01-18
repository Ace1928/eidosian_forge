from typing import (
from .constants import (
from .exceptions import (
from .utils import (
def on_unregistered(self) -> None:
    """
        Called by ``cmd2.Cmd`` after a CommandSet has been unregistered and all its commands removed from the CLI.
        Subclasses can override this to perform remaining cleanup steps.
        """
    self._cmd = None