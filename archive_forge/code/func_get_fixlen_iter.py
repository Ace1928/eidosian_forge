import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
def get_fixlen_iter(self, start=0):
    for i in range(start, self.data.size(0) - 1, self.bptt):
        yield self.get_batch(i)