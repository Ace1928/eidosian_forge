import warnings
import networkx as nx
from networkx.utils import open_file
def make_qstr(t):
    """Returns the string representation of t.
    Add outer double-quotes if the string has a space.
    """
    if not isinstance(t, str):
        t = str(t)
    if ' ' in t:
        t = f'"{t}"'
    return t