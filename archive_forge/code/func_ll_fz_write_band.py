from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def ll_fz_write_band(writer, stride, band_height, samples):
    """
    Low-level wrapper for `::fz_write_band()`.
    Cause a band writer to write the next band
    of data for an image.

    stride: The byte offset from the first byte of the data
    for a pixel to the first byte of the data for the same pixel
    on the row below.

    band_height: The number of lines in this band.

    samples: Pointer to first byte of the data.
    """
    return _mupdf.ll_fz_write_band(writer, stride, band_height, samples)