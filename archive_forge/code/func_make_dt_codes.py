from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
def make_dt_codes(codes_seqs: ty.Sequence[ty.Sequence]) -> Recoder:
    """Create full dt codes Recoder instance from datatype codes

    Include created numpy dtype (from numpy type) and opposite endian
    numpy dtype

    Parameters
    ----------
    codes_seqs : sequence of sequences
       contained sequences make be length 3 or 4, but must all be the same
       length. Elements are data type code, data type name, and numpy
       type (such as ``np.float32``).  The fourth element is the nifti string
       representation of the code (e.g. "NIFTI_TYPE_FLOAT32")

    Returns
    -------
    rec : ``Recoder`` instance
       Recoder that, by default, returns ``code`` when indexed with any
       of the corresponding code, name, type, dtype, or swapped dtype.
       You can also index with ``niistring`` values if codes_seqs had sequences
       of length 4 instead of 3.
    """
    fields = ['code', 'label', 'type']
    len0 = len(codes_seqs[0])
    if len0 not in (3, 4):
        raise ValueError('Sequences must be length 3 or 4')
    if len0 == 4:
        fields.append('niistring')
    dt_codes = []
    for seq in codes_seqs:
        if len(seq) != len0:
            raise ValueError('Sequences must all have the same length')
        np_type = seq[2]
        this_dt = np.dtype(np_type)
        code_syns = list(seq) + [this_dt, this_dt.newbyteorder(swapped_code)]
        dt_codes.append(code_syns)
    return Recoder(dt_codes, fields + ['dtype', 'sw_dtype'], DtypeMapper)