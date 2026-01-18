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
def fz_parse_pdfocr_options(self, args):
    """
        Class-aware wrapper for `::fz_parse_pdfocr_options()`.
        	Parse PDFOCR options.

        	Currently defined options and values are as follows:

        		compression=none: No compression
        		compression=flate: Flate compression
        		strip-height=n: Strip height (default 16)
        		ocr-language=<lang>: OCR Language (default eng)
        		ocr-datadir=<datadir>: OCR data path (default rely on TESSDATA_PREFIX)
        """
    return _mupdf.FzPdfocrOptions_fz_parse_pdfocr_options(self, args)