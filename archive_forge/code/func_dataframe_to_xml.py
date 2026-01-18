import pickle
from _pydevd_bundle.pydevd_constants import get_frame, get_current_thread_id, \
from _pydevd_bundle.pydevd_xml import ExceptionOnEvaluate, get_type, var_to_xml
from _pydev_bundle import pydev_log
import functools
from _pydevd_bundle.pydevd_thread_lifecycle import resume_threads, mark_thread_suspended, suspend_all_threads
from _pydevd_bundle.pydevd_comm_constants import CMD_SET_BREAK
import sys  # @Reimport
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_save_locals, pydevd_timeout, pydevd_constants
from _pydev_bundle.pydev_imports import Exec, execfile
from _pydevd_bundle.pydevd_utils import to_string
import inspect
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_save_locals import update_globals_and_locals
from functools import lru_cache
def dataframe_to_xml(df, name, roffset, coffset, rows, cols, format):
    """
    :type df: pandas.core.frame.DataFrame
    :type name: str
    :type coffset: int
    :type roffset: int
    :type rows: int
    :type cols: int
    :type format: str


    """
    num_rows = min(df.shape[0], MAX_SLICE_SIZE)
    num_cols = min(df.shape[1], MAX_SLICE_SIZE)
    if (num_rows, num_cols) != df.shape:
        df = df.iloc[0:num_rows, 0:num_cols]
        slice = '.iloc[0:%s, 0:%s]' % (num_rows, num_cols)
    else:
        slice = ''
    slice = name + slice
    xml = '<array slice="%s" rows="%s" cols="%s" format="" type="" max="0" min="0"/>\n' % (slice, num_rows, num_cols)
    if (rows, cols) == (-1, -1):
        rows, cols = (num_rows, num_cols)
    rows = min(rows, MAXIMUM_ARRAY_SIZE)
    cols = min(min(cols, MAXIMUM_ARRAY_SIZE), num_cols)
    col_bounds = [None] * cols
    for col in range(cols):
        dtype = df.dtypes.iloc[coffset + col].kind
        if dtype in 'biufc':
            cvalues = df.iloc[:, coffset + col]
            bounds = (cvalues.min(), cvalues.max())
        else:
            bounds = (0, 0)
        col_bounds[col] = bounds
    df = df.iloc[roffset:roffset + rows, coffset:coffset + cols]
    rows, cols = df.shape
    xml += '<headerdata rows="%s" cols="%s">\n' % (rows, cols)
    format = format.replace('%', '')
    col_formats = []
    get_label = lambda label: str(label) if not isinstance(label, tuple) else '/'.join(map(str, label))
    for col in range(cols):
        dtype = df.dtypes.iloc[col].kind
        if dtype == 'f' and format:
            fmt = format
        elif dtype == 'f':
            fmt = '.5f'
        elif dtype == 'i' or dtype == 'u':
            fmt = 'd'
        else:
            fmt = 's'
        col_formats.append('%' + fmt)
        bounds = col_bounds[col]
        xml += '<colheader index="%s" label="%s" type="%s" format="%s" max="%s" min="%s" />\n' % (str(col), get_label(df.axes[1].values[col]), dtype, fmt, bounds[1], bounds[0])
    for row, label in enumerate(iter(df.axes[0])):
        xml += '<rowheader index="%s" label = "%s"/>\n' % (str(row), get_label(label))
    xml += '</headerdata>\n'
    xml += '<arraydata rows="%s" cols="%s"/>\n' % (rows, cols)
    for row in range(rows):
        xml += '<row index="%s"/>\n' % str(row)
        for col in range(cols):
            value = df.iat[row, col]
            value = col_formats[col] % value
            xml += var_to_xml(value, '')
    return xml