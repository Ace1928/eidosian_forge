import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
def read_unknown_ntwk(ntwk):
    if not isinstance(ntwk, nx.classes.graph.Graph):
        _, _, ext = split_filename(ntwk)
        if ext == '.pck':
            ntwk = _read_pickle(ntwk)
        elif ext == '.graphml':
            ntwk = nx.read_graphml(ntwk)
    return ntwk