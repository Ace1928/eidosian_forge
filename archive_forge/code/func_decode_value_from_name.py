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
def decode_value_from_name(encoded_name):
    """Decodes a value that has been encoded in the last bytes of a string.

    Check :func:`encode_value_in_name` to see how the value has been encoded.

    Parameters
    ----------
    encoded_name : bytes
        Name in which a value has been encoded or not.

    Returns
    -------
    name : bytes
        Name without the encoded value.
    value : int
        Value decoded from the name.
    """
    encoded_name = encoded_name.decode('latin1')
    if len(encoded_name) == 0:
        return (encoded_name, 0)
    splits = encoded_name.rstrip('\x00').split('\x00')
    name = splits[0]
    value = 1
    if len(splits) == 2:
        value = int(splits[1])
    elif len(splits) > 2:
        msg = f"Wrong scalar_name or property_name: '{encoded_name}'. Unused characters should be \\x00."
        raise HeaderError(msg)
    return (name, value)