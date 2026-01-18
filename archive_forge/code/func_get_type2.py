import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
def get_type2(path, follow=True):
    """Find the MIMEtype of a file using the XDG recommended checking order.
    
    This first checks the filename, then uses file contents if the name doesn't
    give an unambiguous MIMEtype. It can also handle special filesystem objects
    like directories and sockets.
    
    :param path: file path to examine (need not exist)
    :param follow: whether to follow symlinks
    
    :rtype: :class:`MIMEtype`
    
    .. versionadded:: 1.0
    """
    update_cache()
    try:
        st = os.stat(path) if follow else os.lstat(path)
    except OSError:
        return get_type_by_name(path) or octet_stream
    if not stat.S_ISREG(st.st_mode):
        return _get_type_by_stat(st.st_mode)
    mtypes = sorted(globs.all_matches(path), key=lambda x: x[1], reverse=True)
    if mtypes:
        max_weight = mtypes[0][1]
        i = 1
        for mt, w in mtypes[1:]:
            if w < max_weight:
                break
            i += 1
        mtypes = mtypes[:i]
        if len(mtypes) == 1:
            return mtypes[0][0]
        possible = [mt for mt, w in mtypes]
    else:
        possible = None
    try:
        t = magic.match(path, possible=possible)
    except IOError:
        t = None
    if t:
        return t
    elif mtypes:
        return mtypes[0][0]
    elif stat.S_IMODE(st.st_mode) & 73:
        return app_exe
    else:
        return text if is_text_file(path) else octet_stream