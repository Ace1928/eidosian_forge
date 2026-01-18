import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
def compute_edge_measures(ntwk):
    """
    These return edge-based measures
    """
    iflogger.info('Computing edge measures:')
    measures = {}
    return measures