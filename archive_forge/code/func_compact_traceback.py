import select
import socket
import sys
import time
import warnings
import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, EINVAL, \
def compact_traceback():
    t, v, tb = sys.exc_info()
    tbinfo = []
    if not tb:
        raise AssertionError('traceback does not exist')
    while tb:
        tbinfo.append((tb.tb_frame.f_code.co_filename, tb.tb_frame.f_code.co_name, str(tb.tb_lineno)))
        tb = tb.tb_next
    del tb
    file, function, line = tbinfo[-1]
    info = ' '.join(['[%s|%s|%s]' % x for x in tbinfo])
    return ((file, function, line), t, v, info)