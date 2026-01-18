import json
import math
from collections import defaultdict
from typing import Union, DefaultDict, Dict, List, Optional
import torch
from pydantic import BaseModel
from outlines.fsm.fsm import RegexFSM
from outlines.fsm.json_schema import build_regex_from_schema
def convert_token_to_string(token: str) -> str:
    from transformers.file_utils import SPIECE_UNDERLINE
    string = tokenizer.convert_tokens_to_string([token])
    if token.startswith(SPIECE_UNDERLINE) or token == '<0x20>':
        return ' ' + string
    return string