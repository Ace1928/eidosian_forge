import os
import string
import struct
import warnings
import numpy as np
import nibabel as nib
from nibabel.openers import Opener
from nibabel.orientations import aff2axcodes, axcodes2ornt
from nibabel.volumeutils import endian_codes, native_code, swapped_code
from .array_sequence import create_arraysequences_from_generator
from .header import Field
from .tractogram import LazyTractogram, Tractogram, TractogramItem
from .tractogram_file import DataError, HeaderError, HeaderWarning, TractogramFile
from .utils import peek_next
def encode_value_in_name(value, name, max_name_len=20):
    """Return `name` as fixed-length string, appending `value` as string.

    Form output from `name` if `value <= 1` else `name` + ``\x00`` +
    str(value).

    Return output as fixed length string length `max_name_len`, padded with
    ``\x00``.

    This function also verifies that the modified length of name is less than
    `max_name_len`.

    Parameters
    ----------
    value : int
        Integer value to encode.
    name : str
        Name to which we may append an ascii / latin-1 representation of
        `value`.
    max_name_len : int, optional
        Maximum length of byte string that output can have.

    Returns
    -------
    encoded_name : bytes
        Name maybe followed by ``\x00`` and ascii / latin-1 representation of
        `value`, padded with ``\x00`` bytes.
    """
    if len(name) > max_name_len:
        msg = f"Data information named '{name}' is too long (max {max_name_len} characters.)"
        raise ValueError(msg)
    encoded_name = name if value <= 1 else name + '\x00' + str(value)
    if len(encoded_name) > max_name_len:
        msg = f"Data information named '{name}' is too long (need to be less than {max_name_len - (len(str(value)) + 1)} characters when storing more than one value for a given data information."
        raise ValueError(msg)
    return encoded_name.ljust(max_name_len, '\x00').encode('latin1')