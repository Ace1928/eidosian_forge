import errno
import json
import operator
import os
import shutil
import site
from optparse import SUPPRESS_HELP, Values
from typing import List, Optional
from pip._vendor.rich import print_json
from pip._internal.cache import WheelCache
from pip._internal.cli import cmdoptions
from pip._internal.cli.cmdoptions import make_target_python
from pip._internal.cli.req_command import (
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.exceptions import CommandError, InstallationError
from pip._internal.locations import get_scheme
from pip._internal.metadata import get_environment
from pip._internal.models.installation_report import InstallationReport
from pip._internal.operations.build.build_tracker import get_build_tracker
from pip._internal.operations.check import ConflictDetails, check_install_conflicts
from pip._internal.req import install_given_reqs
from pip._internal.req.req_install import (
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.filesystem import test_writable_dir
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import (
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.virtualenv import (
from pip._internal.wheel_builder import build, should_build_for_install_command
def create_os_error_message(error: OSError, show_traceback: bool, using_user_site: bool) -> str:
    """Format an error message for an OSError

    It may occur anytime during the execution of the install command.
    """
    parts = []
    parts.append('Could not install packages due to an OSError')
    if not show_traceback:
        parts.append(': ')
        parts.append(str(error))
    else:
        parts.append('.')
    parts[-1] += '\n'
    if error.errno == errno.EACCES:
        user_option_part = 'Consider using the `--user` option'
        permissions_part = 'Check the permissions'
        if not running_under_virtualenv() and (not using_user_site):
            parts.extend([user_option_part, ' or ', permissions_part.lower()])
        else:
            parts.append(permissions_part)
        parts.append('.\n')
    if WINDOWS and error.errno == errno.ENOENT and error.filename and (len(error.filename) > 260):
        parts.append('HINT: This error might have occurred since this system does not have Windows Long Path support enabled. You can find information on how to enable this at https://pip.pypa.io/warnings/enable-long-paths\n')
    return ''.join(parts).strip() + '\n'