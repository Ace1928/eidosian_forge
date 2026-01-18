from parlai.core.opt import Opt
from parlai.core.build_data import modelzoo_path
from parlai.utils.bpe import bpe_factory, BPEHelper
from .agents import Agent
from .build_data import make_dir
from collections import defaultdict
import codecs
import copy
import numpy as np
import os
import json
import re
import parlai.utils.logging as logging
from typing import List
def span_tokenize(self, text):
    """
        Tokenize and find  starting index of each token in the original string.
        """
    tokens = self.tokenize(text)
    curr_idx = 0
    indices = []
    for t in tokens:
        while text[curr_idx] != t[0]:
            curr_idx += 1
        indices.append((curr_idx, curr_idx + len(t)))
        curr_idx += len(t)
    return (tokens, indices)