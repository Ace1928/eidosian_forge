import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
def count_sents(self, sents, verbose=False):
    """
        sents : a list of sentences, each a list of tokenized symbols
        """
    if verbose:
        logger.info(f'counting {len(sents)} sents ...')
    for idx, symbols in enumerate(sents):
        if verbose and idx > 0 and (idx % 500000 == 0):
            logger.info(f'    line {idx}')
        self.counter.update(symbols)