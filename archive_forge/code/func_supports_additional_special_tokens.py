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
def supports_additional_special_tokens(self):
    """
        Indicates whether the dictionary supports additional special tokens.
        """
    return self.tokenizer in ['bytelevelbpe', 'split', 'space']