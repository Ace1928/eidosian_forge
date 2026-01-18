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
def load_json(path: str) -> Union[Dict, List]:
    with open(path, 'r') as f:
        return json.load(f)