import argparse
import json
import os
import re
import sys
import types
import torch
from transformers import AutoTokenizer, GPT2Config
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint
def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.

    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split('.')
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d