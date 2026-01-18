import itertools
import random
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple
from ..util import registry
from .example import Example
from .iob_utils import _doc_to_biluo_tags_with_partial, split_bilu_label
def orth_variants_augmenter(nlp: 'Language', example: Example, orth_variants: Dict[str, List[Dict]], *, level: float=0.0, lower: float=0.0) -> Iterator[Example]:
    if random.random() >= level:
        yield example
    else:
        raw_text = example.text
        orig_dict = example.to_dict()
        orig_dict['doc_annotation']['entities'] = _doc_to_biluo_tags_with_partial(example.reference)
        variant_text, variant_token_annot = make_orth_variants(nlp, raw_text, orig_dict['token_annotation'], orth_variants, lower=raw_text is not None and random.random() < lower)
        orig_dict['token_annotation'] = variant_token_annot
        yield example.from_dict(nlp.make_doc(variant_text), orig_dict)