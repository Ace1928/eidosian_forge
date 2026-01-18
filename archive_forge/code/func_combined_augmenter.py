import itertools
import random
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple
from ..util import registry
from .example import Example
from .iob_utils import _doc_to_biluo_tags_with_partial, split_bilu_label
def combined_augmenter(nlp: 'Language', example: Example, *, lower_level: float=0.0, orth_level: float=0.0, orth_variants: Optional[Dict[str, List[Dict]]]=None, whitespace_level: float=0.0, whitespace_per_token: float=0.0, whitespace_variants: Optional[List[str]]=None) -> Iterator[Example]:
    if random.random() < lower_level:
        example = make_lowercase_variant(nlp, example)
    if orth_variants and random.random() < orth_level:
        raw_text = example.text
        orig_dict = example.to_dict()
        orig_dict['doc_annotation']['entities'] = _doc_to_biluo_tags_with_partial(example.reference)
        variant_text, variant_token_annot = make_orth_variants(nlp, raw_text, orig_dict['token_annotation'], orth_variants, lower=False)
        orig_dict['token_annotation'] = variant_token_annot
        example = example.from_dict(nlp.make_doc(variant_text), orig_dict)
    if whitespace_variants and random.random() < whitespace_level:
        for _ in range(int(len(example.reference) * whitespace_per_token)):
            example = make_whitespace_variant(nlp, example, random.choice(whitespace_variants), random.randrange(0, len(example.reference)))
    yield example