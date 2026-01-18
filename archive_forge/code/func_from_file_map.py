from http://www.nitrc.org/projects/gifti/
from __future__ import annotations
import base64
import sys
import warnings
from copy import copy
from typing import Type, cast
import numpy as np
from .. import xmlutils as xml
from ..caret import CaretMetaData
from ..deprecated import deprecate_with_version
from ..filebasedimages import SerializableImage
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from .util import KIND2FMT, array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
from .parse_gifti_fast import GiftiImageParser
@classmethod
def from_file_map(klass, file_map, buffer_size=35000000, mmap=True):
    """Load a Gifti image from a file_map

        Parameters
        ----------
        file_map : dict
            Dictionary with single key ``image`` with associated value which is
            a :class:`FileHolder` instance pointing to the image file.

        buffer_size: None or int, optional
            size of read buffer. None uses default buffer_size
            from xml.parsers.expat.

        mmap : {True, False, 'c', 'r', 'r+'}
            Controls the use of numpy memory mapping for reading data.  Only
            has an effect when loading GIFTI images with data stored in
            external files (``DataArray`` elements with an ``Encoding`` equal
            to ``ExternalFileBinary``).  If ``False``, do not try numpy
            ``memmap`` for data array.  If one of ``{'c', 'r', 'r+'}``, try
            numpy ``memmap`` with ``mode=mmap``.  A `mmap` value of ``True``
            gives the same behavior as ``mmap='c'``.  If the file cannot be
            memory-mapped, ignore `mmap` value and read array from file.

        Returns
        -------
        img : GiftiImage
        """
    parser = klass.parser(buffer_size=buffer_size, mmap=mmap)
    with file_map['image'].get_prepare_fileobj('rb') as fptr:
        parser.parse(fptr=fptr)
    return parser.img