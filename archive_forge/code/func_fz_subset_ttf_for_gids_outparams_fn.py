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
def fz_subset_ttf_for_gids_outparams_fn(orig, num_gids, symbolic, cidfont):
    """
    Class-aware helper for out-params of fz_subset_ttf_for_gids() [fz_subset_ttf_for_gids()].
    """
    ret, gids = ll_fz_subset_ttf_for_gids(orig.m_internal, num_gids, symbolic, cidfont)
    return (FzBuffer(ll_fz_keep_buffer(ret)), gids)