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
def fz_write_band(self, stride, band_height, samples):
    """
        Class-aware wrapper for `::fz_write_band()`.
        	Cause a band writer to write the next band
        	of data for an image.

        	stride: The byte offset from the first byte of the data
        	for a pixel to the first byte of the data for the same pixel
        	on the row below.

        	band_height: The number of lines in this band.

        	samples: Pointer to first byte of the data.
        """
    return _mupdf.FzBandWriter_fz_write_band(self, stride, band_height, samples)