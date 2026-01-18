from __future__ import annotations
import io
import typing as ty
from copy import copy
from .openers import ImageOpener
def copy_file_map(file_map: FileMap) -> FileMap:
    """Copy mapping of fileholders given by `file_map`

    Parameters
    ----------
    file_map : mapping
       mapping of ``FileHolder`` instances

    Returns
    -------
    fm_copy : dict
       Copy of `file_map`, using shallow copy of ``FileHolder``\\s

    """
    return {key: copy(fh) for key, fh in file_map.items()}