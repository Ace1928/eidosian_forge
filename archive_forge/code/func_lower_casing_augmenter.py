import itertools
import random
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple
from ..util import registry
from .example import Example
from .iob_utils import _doc_to_biluo_tags_with_partial, split_bilu_label
def lower_casing_augmenter(nlp: 'Language', example: Example, *, level: float) -> Iterator[Example]:
    if random.random() >= level:
        yield example
    else:
        yield make_lowercase_variant(nlp, example)