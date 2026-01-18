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
def fz_run_display_list(self, dev, ctm, scissor, cookie):
    """
        Class-aware wrapper for `::fz_run_display_list()`.
        	(Re)-run a display list through a device.

        	list: A display list, created by fz_new_display_list and
        	populated with objects from a page by running fz_run_page on a
        	device obtained from fz_new_list_device.

        	ctm: Transform to apply to display list contents. May include
        	for example scaling and rotation, see fz_scale, fz_rotate and
        	fz_concat. Set to fz_identity if no transformation is desired.

        	scissor: Only the part of the contents of the display list
        	visible within this area will be considered when the list is
        	run through the device. This does not imply for tile objects
        	contained in the display list.

        	cookie: Communication mechanism between caller and library
        	running the page. Intended for multi-threaded applications,
        	while single-threaded applications set cookie to NULL. The
        	caller may abort an ongoing page run. Cookie also communicates
        	progress information back to the caller. The fields inside
        	cookie are continually updated while the page is being run.
        """
    return _mupdf.FzDisplayList_fz_run_display_list(self, dev, ctm, scissor, cookie)