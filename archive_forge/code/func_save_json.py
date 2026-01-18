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
def save_json(data, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)