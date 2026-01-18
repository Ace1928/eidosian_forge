import re
import copy
import json
import warnings
from .symbol import Symbol
def looks_like_weight(name):
    """Internal helper to figure out if node should be hidden with `hide_weights`.
        """
    weight_like = ('_weight', '_bias', '_beta', '_gamma', '_moving_var', '_moving_mean', '_running_var', '_running_mean')
    return name.endswith(weight_like)