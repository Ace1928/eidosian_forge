from __future__ import annotations
import sys
from io import StringIO
from json import JSONEncoder, loads
from typing import TYPE_CHECKING
def pprint_json(data):
    """
    Display a tree-like object in a jupyter notebook.
    Allows for collapsible interactive interaction with data.

    Args:
        data: a dictionary or object

    Based on:
    https://gist.github.com/jmmshn/d37d5a1be80a6da11f901675f195ca22

    """
    from IPython.display import JSON, display
    display(JSON(loads(DisplayEcoder().encode(data))))