import logging
import os
import shutil
import stat
import tarfile
import zipfile
from typing import Iterable, List, Optional
from zipfile import ZipInfo
from pip._internal.exceptions import InstallationError
from pip._internal.utils.filetypes import (
from pip._internal.utils.misc import ensure_dir
def set_extracted_file_to_default_mode_plus_executable(path: str) -> None:
    """
    Make file present at path have execute for user/group/world
    (chmod +x) is no-op on windows per python docs
    """
    os.chmod(path, 511 & ~current_umask() | 73)