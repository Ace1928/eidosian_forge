import json
import os
import warnings
from functools import lru_cache
from typing import List, Optional, Tuple, Union
import numpy as np
import regex as re
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
from .english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
def new_chunk():
    return {'language': last_language, 'timestamp': [None, None], 'text': ''}