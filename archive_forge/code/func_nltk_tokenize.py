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
def nltk_tokenize(self, text, building=False):
    """
        Tokenize using NLTK PunktTokenizer.

        Uses nltk-trained PunktTokenizer for sentence tokenization and Treebank Word
        Tokenizer for tokenizing words within sentences.
        """
    return (token for sent in self.sent_tok.tokenize(text) for token in self.word_tok.tokenize(sent))