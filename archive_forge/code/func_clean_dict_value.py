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
def clean_dict_value(value):
    if not isinstance(value, list):
        return value
    if len(value) == 1:
        return value[0]
    if value[0] == LIST_START_VALUE:
        return value[1:]
    return value