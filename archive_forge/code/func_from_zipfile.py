import email.message
import importlib.metadata
import os
import pathlib
import zipfile
from typing import (
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.exceptions import InvalidWheel, UnsupportedWheel
from pip._internal.metadata.base import (
from pip._internal.utils.misc import normalize_path
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.wheel import parse_wheel, read_wheel_metadata_file
from ._compat import BasePath, get_dist_name
@classmethod
def from_zipfile(cls, zf: zipfile.ZipFile, name: str, location: str) -> 'WheelDistribution':
    info_dir, _ = parse_wheel(zf, name)
    paths = ((name, pathlib.PurePosixPath(name.split('/', 1)[-1])) for name in zf.namelist() if name.startswith(f'{info_dir}/'))
    files = {relpath: read_wheel_metadata_file(zf, fullpath) for fullpath, relpath in paths}
    info_location = pathlib.PurePosixPath(location, info_dir)
    return cls(files, info_location)