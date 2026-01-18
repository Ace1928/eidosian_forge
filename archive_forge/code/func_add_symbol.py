import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
def add_symbol(self, sym):
    if sym not in self.sym2idx:
        self.idx2sym.append(sym)
        self.sym2idx[sym] = len(self.idx2sym) - 1