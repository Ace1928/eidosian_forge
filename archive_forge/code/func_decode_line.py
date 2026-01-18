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
def decode_line(line):
    if isinstance(line, bytes):
        try:
            line.decode('ascii')
        except UnicodeDecodeError as err:
            raise NetworkXError('input is not ASCII-encoded') from err
    if not isinstance(line, str):
        line = str(line)
    return line