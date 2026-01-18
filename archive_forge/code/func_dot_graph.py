from __future__ import annotations
import os
import re
from functools import partial
from dask.core import get_dependencies, ishashable, istask
from dask.utils import apply, funcname, import_required, key_split
def dot_graph(dsk, filename='mydask', format=None, **kwargs):
    """
    Render a task graph using dot.

    If `filename` is not None, write a file to disk with the specified name and extension.
    If no extension is specified, '.png' will be used by default.

    Parameters
    ----------
    dsk : dict
        The graph to display.
    filename : str or None, optional
        The name of the file to write to disk. If the provided `filename`
        doesn't include an extension, '.png' will be used by default.
        If `filename` is None, no file will be written, and we communicate
        with dot using only pipes.  Default is 'mydask'.
    format : {'png', 'pdf', 'dot', 'svg', 'jpeg', 'jpg'}, optional
        Format in which to write output file.  Default is 'png'.
    **kwargs
        Additional keyword arguments to forward to `to_graphviz`.

    Returns
    -------
    result : None or IPython.display.Image or IPython.display.SVG  (See below.)

    Notes
    -----
    If IPython is installed, we return an IPython.display object in the
    requested format.  If IPython is not installed, we just return None.

    We always return None if format is 'pdf' or 'dot', because IPython can't
    display these formats natively. Passing these formats with filename=None
    will not produce any useful output.

    See Also
    --------
    dask.dot.to_graphviz
    """
    g = to_graphviz(dsk, **kwargs)
    return graphviz_to_file(g, filename, format)