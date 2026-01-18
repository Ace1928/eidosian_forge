import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece
from ...tokenization_utils import BatchEncoding, PreTrainedTokenizer
from ...utils import logging
def get_lang_id(self, lang: str) -> int:
    lang_token = self.get_lang_token(lang)
    return self.lang_token_to_id[lang_token]