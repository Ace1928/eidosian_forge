import itertools
import random
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple
from ..util import registry
from .example import Example
from .iob_utils import _doc_to_biluo_tags_with_partial, split_bilu_label
@registry.augmenters('spacy.lower_case.v1')
def create_lower_casing_augmenter(level: float) -> Callable[['Language', Example], Iterator[Example]]:
    """Create a data augmentation callback that converts documents to lowercase.
    The callback can be added to a corpus or other data iterator during training.

    level (float): The percentage of texts that will be augmented.
    RETURNS (Callable[[Language, Example], Iterator[Example]]): The augmenter.
    """
    return partial(lower_casing_augmenter, level=level)