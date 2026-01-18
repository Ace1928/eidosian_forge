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
def from_metadata_file_contents(cls, metadata_contents: bytes, filename: str, project_name: str) -> BaseDistribution:
    temp_dir = pathlib.Path(TempDirectory(kind='metadata', globally_managed=True).path)
    metadata_path = temp_dir / 'METADATA'
    metadata_path.write_bytes(metadata_contents)
    dist = importlib.metadata.Distribution.at(metadata_path.parent)
    return cls(dist, metadata_path.parent, None)