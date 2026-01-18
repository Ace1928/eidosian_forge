from typing import (
from .constants import (
from .exceptions import (
from .utils import (
def on_register(self, cmd: 'cmd2.Cmd') -> None:
    """
        Called by cmd2.Cmd as the first step to registering a CommandSet. The commands defined in this class have
        not been added to the CLI object at this point. Subclasses can override this to perform any initialization
        requiring access to the Cmd object (e.g. configure commands and their parsers based on CLI state data).

        :param cmd: The cmd2 main application
        """
    if self._cmd is None:
        self._cmd = cmd
    else:
        raise CommandSetRegistrationError('This CommandSet has already been registered')