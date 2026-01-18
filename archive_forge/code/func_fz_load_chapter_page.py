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
def fz_load_chapter_page(self, chapter, page):
    """
        Class-aware wrapper for `::fz_load_chapter_page()`.
        	Load a page.

        	After fz_load_page is it possible to retrieve the size of the
        	page using fz_bound_page, or to render the page using
        	fz_run_page_*. Free the page by calling fz_drop_page.

        	chapter: chapter number, 0 is the first chapter of the document.
        	number: page number, 0 is the first page of the chapter.
        """
    return _mupdf.FzDocument_fz_load_chapter_page(self, chapter, page)