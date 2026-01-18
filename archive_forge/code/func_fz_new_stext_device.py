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
def fz_new_stext_device(self, options):
    """
        Class-aware wrapper for `::fz_new_stext_device()`.
        	Create a device to extract the text on a page.

        	Gather the text on a page into blocks and lines.

        	The reading order is taken from the order the text is drawn in
        	the source file, so may not be accurate.

        	page: The text page to which content should be added. This will
        	usually be a newly created (empty) text page, but it can be one
        	containing data already (for example when merging multiple
        	pages, or watermarking).

        	options: Options to configure the stext device.
        """
    return _mupdf.FzStextPage_fz_new_stext_device(self, options)