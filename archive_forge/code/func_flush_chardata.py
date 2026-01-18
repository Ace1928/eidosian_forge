import base64
import os.path as op
import sys
import warnings
import zlib
from io import StringIO
from xml.parsers.expat import ExpatError
import numpy as np
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from ..xmlutils import XmlParser
from .gifti import (
from .util import array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
def flush_chardata(self):
    """Collate and process collected character data"""
    if self.write_to != 'Data' and self._char_blocks is None:
        return
    if self._char_blocks is not None:
        data = ''.join(self._char_blocks)
    else:
        data = None
    self._char_blocks = None
    if self.write_to == 'Name':
        data = data.strip()
        self.nvpair[0] = data
    elif self.write_to == 'Value':
        data = data.strip()
        self.nvpair[1] = data
    elif self.write_to == 'DataSpace':
        data = data.strip()
        self.coordsys.dataspace = xform_codes.code[data]
    elif self.write_to == 'TransformedSpace':
        data = data.strip()
        self.coordsys.xformspace = xform_codes.code[data]
    elif self.write_to == 'MatrixData':
        c = StringIO(data)
        self.coordsys.xform = np.loadtxt(c)
        c.close()
    elif self.write_to == 'Data':
        self.da.data = read_data_block(self.da, self.fname, data, self.mmap)
        self.endian = gifti_endian_codes.code[sys.byteorder]
    elif self.write_to == 'Label':
        self.label.label = data.strip()