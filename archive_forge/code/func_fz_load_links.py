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
def fz_load_links(self):
    """
        Class-aware wrapper for `::fz_load_links()`.
        	Load the list of links for a page.

        	Returns a linked list of all the links on the page, each with
        	its clickable region and link destination. Each link is
        	reference counted so drop and free the list of links by
        	calling fz_drop_link on the pointer return from fz_load_links.

        	page: Page obtained from fz_load_page.
        """
    return _mupdf.FzPage_fz_load_links(self)