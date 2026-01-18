import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as sp
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
def save_pretrained(self, path: str, filename_prefix: str=None):
    filename = VOCAB_FILES_NAMES[list(VOCAB_FILES_NAMES.keys())[0]]
    if filename_prefix is not None:
        filename = filename_prefix + '-' + filename
    full_path = os.path.join(path, filename)
    with open(full_path, 'wb') as fs:
        fs.write(self.spm.serialized_model_proto())
    return (full_path,)