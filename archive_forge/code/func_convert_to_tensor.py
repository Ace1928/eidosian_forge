import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
@torch_only_method
def convert_to_tensor(self, symbols):
    return torch.LongTensor(self.convert_tokens_to_ids(symbols))