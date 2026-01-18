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
def fz_aes_setkey_dec(self, key, keysize):
    """
        Class-aware wrapper for `::fz_aes_setkey_dec()`.
        	AES decryption intialisation. Fills in the supplied context
        	and prepares for decryption using the given key.

        	Returns non-zero for error (key size other than 128/192/256).

        	Never throws an exception.
        """
    return _mupdf.FzAes_fz_aes_setkey_dec(self, key, keysize)