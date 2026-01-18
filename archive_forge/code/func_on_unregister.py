from typing import (
from .constants import (
from .exceptions import (
from .utils import (
def on_unregister(self) -> None:
    """
        Called by ``cmd2.Cmd`` as the first step to unregistering a CommandSet. Subclasses can override this to
        perform any cleanup steps which require their commands being registered in the CLI.
        """
    pass