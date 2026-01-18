from __future__ import annotations
import inspect
import os
from inspect import currentframe, getframeinfo
from typing import TYPE_CHECKING
def find_top_pyfile():
    """
    This function inspects the Cpython frame to find the path of the script.
    """
    frame = currentframe()
    while True:
        if frame.f_back is None:
            finfo = getframeinfo(frame)
            return os.path.abspath(finfo.filename)
        frame = frame.f_back