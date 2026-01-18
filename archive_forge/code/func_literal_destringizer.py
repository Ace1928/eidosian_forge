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
def literal_destringizer(rep):
    """Convert a Python literal to the value it represents.

    Parameters
    ----------
    rep : string
        A Python literal.

    Returns
    -------
    value : object
        The value of the Python literal.

    Raises
    ------
    ValueError
        If `rep` is not a Python literal.
    """
    if isinstance(rep, str):
        orig_rep = rep
        try:
            return literal_eval(rep)
        except SyntaxError as err:
            raise ValueError(f'{orig_rep!r} is not a valid Python literal') from err
    else:
        raise ValueError(f'{rep!r} is not a string')