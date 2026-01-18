import json
import os
import re
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def normalize(self, x: str) -> str:
    """Cover moses empty string edge case. They return empty list for '' input!"""
    return self.punc_normalizer(x) if x else ''