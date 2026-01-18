import sys
import sysconfig
import os
import re
from distutils import log
from distutils.core import Command
from distutils.debug import DEBUG
from distutils.sysconfig import get_config_vars
from distutils.errors import DistutilsPlatformError
from distutils.file_util import write_file
from distutils.util import convert_path, subst_vars, change_root
from distutils.util import get_platform
from distutils.errors import DistutilsOptionError
from site import USER_BASE
from site import USER_SITE
def convert_paths(self, *names):
    """Call `convert_path` over `names`."""
    for name in names:
        attr = 'install_' + name
        setattr(self, attr, convert_path(getattr(self, attr)))