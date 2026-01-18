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
def fz_aes_crypt_cbc(self, mode, length, iv, input, output):
    """
        Class-aware wrapper for `::fz_aes_crypt_cbc()`.
        	AES block processing. Encrypts or Decrypts (according to mode,
        	which must match what was initially set up) length bytes (which
        	must be a multiple of 16), using (and modifying) the insertion
        	vector iv, reading from input, and writing to output.

        	Never throws an exception.
        """
    return _mupdf.FzAes_fz_aes_crypt_cbc(self, mode, length, iv, input, output)