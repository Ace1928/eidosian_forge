import re
from typing import FrozenSet, NewType, Tuple, Union, cast
from .tags import Tag, parse_tag
from .version import InvalidVersion, Version
def parse_sdist_filename(filename: str) -> Tuple[NormalizedName, Version]:
    if filename.endswith('.tar.gz'):
        file_stem = filename[:-len('.tar.gz')]
    elif filename.endswith('.zip'):
        file_stem = filename[:-len('.zip')]
    else:
        raise InvalidSdistFilename(f"Invalid sdist filename (extension must be '.tar.gz' or '.zip'): {filename}")
    name_part, sep, version_part = file_stem.rpartition('-')
    if not sep:
        raise InvalidSdistFilename(f'Invalid sdist filename: {filename}')
    name = canonicalize_name(name_part)
    version = Version(version_part)
    return (name, version)