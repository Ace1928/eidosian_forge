from __future__ import unicode_literals
import distutils.errors
from distutils import log
import errno
import io
import os
import re
import subprocess
import time
import pkg_resources
from pbr import options
from pbr import version
def get_is_release(git_dir):
    return _get_raw_tag_info(git_dir) == 0