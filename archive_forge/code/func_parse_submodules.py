import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def parse_submodules(config: ConfigFile) -> Iterator[Tuple[bytes, bytes, bytes]]:
    """Parse a gitmodules GitConfig file, returning submodules.

    Args:
      config: A `ConfigFile`
    Returns:
      list of tuples (submodule path, url, name),
        where name is quoted part of the section's name.
    """
    for section in config.keys():
        section_kind, section_name = section
        if section_kind == b'submodule':
            try:
                sm_path = config.get(section, b'path')
                sm_url = config.get(section, b'url')
                yield (sm_path, sm_url, section_name)
            except KeyError:
                pass