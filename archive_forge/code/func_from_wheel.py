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
def from_wheel(cls, wheel: Wheel, name: str) -> BaseDistribution:
    try:
        with wheel.as_zipfile() as zf:
            dist = WheelDistribution.from_zipfile(zf, name, wheel.location)
    except zipfile.BadZipFile as e:
        raise InvalidWheel(wheel.location, name) from e
    except UnsupportedWheel as e:
        raise UnsupportedWheel(f'{name} has an invalid wheel, {e}')
    return cls(dist, dist.info_location, pathlib.PurePosixPath(wheel.location))