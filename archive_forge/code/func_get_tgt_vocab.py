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
def get_tgt_vocab(self):
    return dict(self.target_encoder, **self.added_tokens_decoder)