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
def resize_to_max(self, maxtokens):
    """
        Trims the dictionary to the maximum number of tokens.
        """
    if maxtokens >= 0 and len(self.tok2ind) > maxtokens:
        for k in range(maxtokens, len(self.ind2tok)):
            v = self.ind2tok[k]
            del self.ind2tok[k]
            del self.tok2ind[v]
            del self.freq[v]