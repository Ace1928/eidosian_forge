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
def remove_tail(self, min_freq):
    """
        Remove elements below the frequency cutoff from the dictionary.
        """
    to_remove = []
    for token, freq in self.freq.items():
        if freq < min_freq:
            to_remove.append(token)
    for token in to_remove:
        del self.freq[token]
        idx = self.tok2ind.pop(token)
        del self.ind2tok[idx]