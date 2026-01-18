import codecs
import io
import math
import os
import tempfile
from ast import literal_eval
from contextlib import contextmanager
from textwrap import dedent
import pytest
import networkx as nx
from networkx.readwrite.gml import literal_destringizer, literal_stringizer
@contextmanager
def byte_file():
    _file_handle = io.BytesIO()
    yield _file_handle
    _file_handle.seek(0)