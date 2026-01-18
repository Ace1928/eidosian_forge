import os
from packaging.version import Version
from ... import logging
from ..base import CommandLine, CommandLineInputSpec, traits, Undefined, PackageInfo
from ...utils.filemanip import split_filename
def get_custom_path(command, env_dir='NIFTYREGDIR'):
    return os.path.join(os.getenv(env_dir, ''), command)