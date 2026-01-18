import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def iter_instead_of(config: Config, push: bool=False) -> Iterable[Tuple[str, str]]:
    """Iterate over insteadOf / pushInsteadOf values."""
    for section in config.sections():
        if section[0] != b'url':
            continue
        replacement = section[1]
        try:
            needles = list(config.get_multivar(section, 'insteadOf'))
        except KeyError:
            needles = []
        if push:
            try:
                needles += list(config.get_multivar(section, 'pushInsteadOf'))
            except KeyError:
                pass
        for needle in needles:
            assert isinstance(needle, bytes)
            yield (needle.decode('utf-8'), replacement.decode('utf-8'))