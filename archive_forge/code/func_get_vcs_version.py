import logging
import os
import re
from typing import List, Optional, Tuple
from pip._internal.utils.misc import (
from pip._internal.utils.subprocess import CommandArgs, make_command
from pip._internal.vcs.versioncontrol import (
def get_vcs_version(self) -> Tuple[int, ...]:
    """Return the version of the currently installed Subversion client.

        If the version of the Subversion client has already been queried,
        a cached value will be used.

        :return: A tuple containing the parts of the version information or
            ``()`` if the version returned from ``svn`` could not be parsed.
        :raises: BadCommand: If ``svn`` is not installed.
        """
    if self._vcs_version is not None:
        return self._vcs_version
    vcs_version = self.call_vcs_version()
    self._vcs_version = vcs_version
    return vcs_version