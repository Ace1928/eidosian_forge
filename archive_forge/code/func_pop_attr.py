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
def pop_attr(dct, category, attr, i):
    try:
        return dct.pop(attr)
    except KeyError as err:
        raise NetworkXError(f'{category} #{i} has no {attr!r} attribute') from err