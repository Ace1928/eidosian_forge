import os
import time
import sys
import zlib
from io import BytesIO
import warnings
import numpy as np
import scipy.sparse
from ._byteordercodes import native_code, swapped_code
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio5_utils import VarReader5
from ._mio5_params import (MatlabObject, MatlabFunction, MDTYPES, NP_TO_MTYPES,
from ._streams import ZlibInputStream
def read_var_header(self):
    """ Read header, return header, next position

        Header has to define at least .name and .is_global

        Parameters
        ----------
        None

        Returns
        -------
        header : object
           object that can be passed to self.read_var_array, and that
           has attributes .name and .is_global
        next_position : int
           position in stream of next variable
        """
    mdtype, byte_count = self._file_reader.read_full_tag()
    if not byte_count > 0:
        raise ValueError('Did not read any bytes')
    next_pos = self.mat_stream.tell() + byte_count
    if mdtype == miCOMPRESSED:
        stream = ZlibInputStream(self.mat_stream, byte_count)
        self._matrix_reader.set_stream(stream)
        check_stream_limit = self.verify_compressed_data_integrity
        mdtype, byte_count = self._matrix_reader.read_full_tag()
    else:
        check_stream_limit = False
        self._matrix_reader.set_stream(self.mat_stream)
    if not mdtype == miMATRIX:
        raise TypeError('Expecting miMATRIX type here, got %d' % mdtype)
    header = self._matrix_reader.read_header(check_stream_limit)
    return (header, next_pos)