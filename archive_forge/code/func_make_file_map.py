from __future__ import annotations
import io
import typing as ty
from copy import deepcopy
from urllib import request
from ._compression import COMPRESSION_ERRORS
from .fileholders import FileHolder, FileMap
from .filename_parser import TypesFilenamesError, _stringify_path, splitext_addext, types_filenames
from .openers import ImageOpener
@classmethod
def make_file_map(klass, mapping: ty.Mapping[str, str | io.IOBase] | None=None) -> FileMap:
    """Class method to make files holder for this image type

        Parameters
        ----------
        mapping : None or mapping, optional
           mapping with keys corresponding to image file types (such as
           'image', 'header' etc, depending on image class) and values
           that are filenames or file-like.  Default is None

        Returns
        -------
        file_map : dict
           dict with string keys given by first entry in tuples in
           sequence klass.files_types, and values of type FileHolder,
           where FileHolder objects have default values, other than
           those given by `mapping`
        """
    if mapping is None:
        mapping = {}
    file_map = {}
    for key, ext in klass.files_types:
        file_map[key] = FileHolder()
        mapval = mapping.get(key, None)
        if isinstance(mapval, str):
            file_map[key].filename = mapval
        elif hasattr(mapval, 'tell'):
            file_map[key].fileobj = mapval
    return file_map