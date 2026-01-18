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
def fz_bidi_fragment_text(text, textlen, baseDir, callback, arg, flags):
    """
     Class-aware wrapper for `::fz_bidi_fragment_text()`.

    	This function has out-params. Python/C# wrappers look like:
    		`fz_bidi_fragment_text(const uint32_t *text, size_t textlen, ::fz_bidi_fragment_fn *callback, void *arg, int flags)` => ::fz_bidi_direction baseDir

    		Partitions the given Unicode sequence into one or more
    		unidirectional fragments and invokes the given callback
    		function for each fragment.

    		For example, if directionality of text is:
    				0123456789
    				rrlllrrrrr,
    		we'll invoke callback with:
    				&text[0], length == 2
    				&text[2], length == 3
    				&text[5], length == 5

    		:type text: int, in
    :param text:	start of Unicode sequence
        		:type textlen: int, in
    :param textlen:   number of Unicodes to analyse
        		:type baseDir: int, in
    :param baseDir:   direction of paragraph (specify FZ_BIDI_NEUTRAL to force auto-detection)
        		:type callback: ::fz_bidi_fragment_fn, in
    :param callback:  function to be called for each fragment
        		:type arg: void, in
    :param arg:	data to be passed to the callback function
        		:type flags: int, in
    :param flags:     flags to control operation (see fz_bidi_flags above)
    """
    return _mupdf.fz_bidi_fragment_text(text, textlen, baseDir, callback, arg, flags)