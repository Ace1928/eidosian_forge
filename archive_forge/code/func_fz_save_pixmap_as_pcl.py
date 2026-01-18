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
def fz_save_pixmap_as_pcl(self, filename, append, pcl):
    """
        Class-aware wrapper for `::fz_save_pixmap_as_pcl()`.
        	Save an (RGB) pixmap as color PCL.
        """
    return _mupdf.FzPixmap_fz_save_pixmap_as_pcl(self, filename, append, pcl)