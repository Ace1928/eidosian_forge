import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def graph_from_dot_file(path, encoding=None):
    """Load graphs from DOT file at `path`.

    @param path: to DOT file
    @param encoding: as passed to `io.open`.
        For example, `'utf-8'`.

    @return: Graphs that result from parsing.
    @rtype: `list` of `pydot.Dot`
    """
    with io.open(path, 'rt', encoding=encoding) as f:
        s = f.read()
    graphs = graph_from_dot_data(s)
    return graphs