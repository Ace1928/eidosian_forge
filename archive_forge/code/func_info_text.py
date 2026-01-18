from collections.abc import Sequence
from traits import __version__ as traits_version
import traits.api as traits
from traits.api import TraitType, Unicode
from traits.trait_base import _Undefined
from pathlib import Path
from ...utils.filemanip import path_resolve
@property
def info_text(self):
    """Create the trait's general description."""
    info_text = 'a pathlike object or string'
    if any((self.exists, self._is_file, self._is_dir)):
        info_text += ' representing a'
        if self.exists:
            info_text += 'n existing'
        if self._is_file:
            info_text += ' file'
        elif self._is_dir:
            info_text += ' directory'
        else:
            info_text += ' file or directory'
    return info_text