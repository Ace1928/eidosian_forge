import configparser
import logging
import os
from typing import List, Optional, Tuple
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import HiddenText, display_path
from pip._internal.utils.subprocess import make_command
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs.versioncontrol import (
@classmethod
def get_repository_root(cls, location: str) -> Optional[str]:
    loc = super().get_repository_root(location)
    if loc:
        return loc
    try:
        r = cls.run_command(['root'], cwd=location, show_stdout=False, stdout_only=True, on_returncode='raise', log_failed_cmd=False)
    except BadCommand:
        logger.debug('could not determine if %s is under hg control because hg is not available', location)
        return None
    except InstallationError:
        return None
    return os.path.normpath(r.rstrip('\r\n'))