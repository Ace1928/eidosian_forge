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
def fz_subset_ttf_for_gids(self, gids, num_gids, symbolic, cidfont):
    """
        Class-aware wrapper for `::fz_subset_ttf_for_gids()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_subset_ttf_for_gids(int num_gids, int symbolic, int cidfont)` => `(fz_buffer *, int gids)`
        """
    return _mupdf.FzBuffer_fz_subset_ttf_for_gids(self, gids, num_gids, symbolic, cidfont)