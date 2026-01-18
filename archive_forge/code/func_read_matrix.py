import warnings
import numpy as np
from scipy.sparse import csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat
def read_matrix(self):
    return _read_hb_data(self._fid, self._hb_info)