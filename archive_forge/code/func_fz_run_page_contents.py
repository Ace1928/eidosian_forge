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
def fz_run_page_contents(self, dev, transform, cookie):
    """
        Class-aware wrapper for `::fz_run_page_contents()`.
        	Run a page through a device. Just the main
        	page content, without the annotations, if any.

        	page: Page obtained from fz_load_page.

        	dev: Device obtained from fz_new_*_device.

        	transform: Transform to apply to page. May include for example
        	scaling and rotation, see fz_scale, fz_rotate and fz_concat.
        	Set to fz_identity if no transformation is desired.

        	cookie: Communication mechanism between caller and library
        	rendering the page. Intended for multi-threaded applications,
        	while single-threaded applications set cookie to NULL. The
        	caller may abort an ongoing rendering of a page. Cookie also
        	communicates progress information back to the caller. The
        	fields inside cookie are continually updated while the page is
        	rendering.
        """
    return _mupdf.FzPage_fz_run_page_contents(self, dev, transform, cookie)