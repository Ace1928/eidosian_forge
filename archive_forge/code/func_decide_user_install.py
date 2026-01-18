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
def decide_user_install(use_user_site: Optional[bool], prefix_path: Optional[str]=None, target_dir: Optional[str]=None, root_path: Optional[str]=None, isolated_mode: bool=False) -> bool:
    """Determine whether to do a user install based on the input options.

    If use_user_site is False, no additional checks are done.
    If use_user_site is True, it is checked for compatibility with other
    options.
    If use_user_site is None, the default behaviour depends on the environment,
    which is provided by the other arguments.
    """
    if use_user_site is not None and (not use_user_site):
        logger.debug('Non-user install by explicit request')
        return False
    if use_user_site:
        if prefix_path:
            raise CommandError("Can not combine '--user' and '--prefix' as they imply different installation locations")
        if virtualenv_no_global():
            raise InstallationError("Can not perform a '--user' install. User site-packages are not visible in this virtualenv.")
        logger.debug('User install by explicit request')
        return True
    assert use_user_site is None
    if prefix_path or target_dir:
        logger.debug('Non-user install due to --prefix or --target option')
        return False
    if not site.ENABLE_USER_SITE:
        logger.debug('Non-user install because user site-packages disabled')
        return False
    if site_packages_writable(root=root_path, isolated=isolated_mode):
        logger.debug('Non-user install because site-packages writeable')
        return False
    logger.info('Defaulting to user installation because normal site-packages is not writeable')
    return True