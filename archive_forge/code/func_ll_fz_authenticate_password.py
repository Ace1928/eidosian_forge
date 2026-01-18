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
def ll_fz_authenticate_password(doc, password):
    """
    Low-level wrapper for `::fz_authenticate_password()`.
    Test if the given password can decrypt the document.

    password: The password string to be checked. Some document
    specifications do not specify any particular text encoding, so
    neither do we.

    Returns 0 for failure to authenticate, non-zero for success.

    For PDF documents, further information can be given by examining
    the bits in the return code.

    	Bit 0 => No password required
    	Bit 1 => User password authenticated
    	Bit 2 => Owner password authenticated
    """
    return _mupdf.ll_fz_authenticate_password(doc, password)