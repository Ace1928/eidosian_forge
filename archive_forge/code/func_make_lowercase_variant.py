import itertools
import random
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple
from ..util import registry
from .example import Example
from .iob_utils import _doc_to_biluo_tags_with_partial, split_bilu_label
def make_lowercase_variant(nlp: 'Language', example: Example):
    example_dict = example.to_dict()
    example_dict['doc_annotation']['entities'] = _doc_to_biluo_tags_with_partial(example.reference)
    doc = nlp.make_doc(example.text.lower())
    example_dict['token_annotation']['ORTH'] = [t.lower_ for t in example.reference]
    return example.from_dict(doc, example_dict)