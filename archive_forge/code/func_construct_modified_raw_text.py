import itertools
import random
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple
from ..util import registry
from .example import Example
from .iob_utils import _doc_to_biluo_tags_with_partial, split_bilu_label
def construct_modified_raw_text(token_dict):
    """Construct modified raw text from words and spaces."""
    raw = ''
    for orth, spacy in zip(token_dict['ORTH'], token_dict['SPACY']):
        raw += orth
        if spacy:
            raw += ' '
    return raw