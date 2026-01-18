import html.entities as htmlentitydefs
import re
import warnings
from ast import literal_eval
from collections import defaultdict
from enum import Enum
from io import StringIO
from typing import Any, NamedTuple
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file
@nx._dispatch(graphs=None)
def parse_gml(lines, label='label', destringizer=None):
    """Parse GML graph from a string or iterable.

    Parameters
    ----------
    lines : string or iterable of strings
       Data in GML format.

    label : string, optional
        If not None, the parsed nodes will be renamed according to node
        attributes indicated by `label`. Default value: 'label'.

    destringizer : callable, optional
        A `destringizer` that recovers values stored as strings in GML. If it
        cannot convert a string to a value, a `ValueError` is raised. Default
        value : None.

    Returns
    -------
    G : NetworkX graph
        The parsed graph.

    Raises
    ------
    NetworkXError
        If the input cannot be parsed.

    See Also
    --------
    write_gml, read_gml

    Notes
    -----
    This stores nested GML attributes as dictionaries in the NetworkX graph,
    node, and edge attribute structures.

    GML files are stored using a 7-bit ASCII encoding with any extended
    ASCII characters (iso8859-1) appearing as HTML character entities.
    Without specifying a `stringizer`/`destringizer`, the code is capable of
    writing `int`/`float`/`str`/`dict`/`list` data as required by the GML
    specification.  For writing other data types, and for reading data other
    than `str` you need to explicitly supply a `stringizer`/`destringizer`.

    For additional documentation on the GML file format, please see the
    `GML url <https://web.archive.org/web/20190207140002/http://www.fim.uni-passau.de/index.php?id=17297&L=1>`_.

    See the module docstring :mod:`networkx.readwrite.gml` for more details.
    """

    def decode_line(line):
        if isinstance(line, bytes):
            try:
                line.decode('ascii')
            except UnicodeDecodeError as err:
                raise NetworkXError('input is not ASCII-encoded') from err
        if not isinstance(line, str):
            line = str(line)
        return line

    def filter_lines(lines):
        if isinstance(lines, str):
            lines = decode_line(lines)
            lines = lines.splitlines()
            yield from lines
        else:
            for line in lines:
                line = decode_line(line)
                if line and line[-1] == '\n':
                    line = line[:-1]
                if line.find('\n') != -1:
                    raise NetworkXError('input line contains newline')
                yield line
    G = parse_gml_lines(filter_lines(lines), label, destringizer)
    return G