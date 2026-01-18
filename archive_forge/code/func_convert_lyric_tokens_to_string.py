import json
import os
import re
import unicodedata
from json.encoder import INFINITY
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import regex
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_flax_available, is_tf_available, is_torch_available, logging
from ...utils.generic import _is_jax, _is_numpy
def convert_lyric_tokens_to_string(self, lyrics: List[str]) -> str:
    return ' '.join(lyrics)