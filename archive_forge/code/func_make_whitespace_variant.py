import itertools
import random
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple
from ..util import registry
from .example import Example
from .iob_utils import _doc_to_biluo_tags_with_partial, split_bilu_label
def make_whitespace_variant(nlp: 'Language', example: Example, whitespace: str, position: int) -> Example:
    """Insert the whitespace token at the specified token offset in the doc.
    This is primarily intended for v2-compatible training data that doesn't
    include links or spans. If the document includes links, spans, or partial
    dependency annotation, it is returned without modifications.

    The augmentation follows the basics of the v2 space attachment policy, but
    without a distinction between "real" and other tokens, so space tokens
    may be attached to space tokens:
    - at the beginning of a sentence attach the space token to the following
      token
    - otherwise attach the space token to the preceding token

    The augmenter does not attempt to consolidate adjacent whitespace in the
    same way that the tokenizer would.

    The following annotation is used for the space token:
    TAG: "_SP"
    MORPH: ""
    POS: "SPACE"
    LEMMA: ORTH
    DEP: "dep"
    SENT_START: False

    The annotation for each attribute is only set for the space token if there
    is already at least partial annotation for that attribute in the original
    example.

    RETURNS (Example): Example with one additional space token.
    """
    example_dict = example.to_dict()
    example_dict['doc_annotation']['entities'] = _doc_to_biluo_tags_with_partial(example.reference)
    doc_dict = example_dict.get('doc_annotation', {})
    token_dict = example_dict.get('token_annotation', {})
    if len(example.reference) == 0 or 'ORTH' not in token_dict or len(doc_dict.get('links', [])) > 0 or (len(example.reference.spans) > 0) or (example.reference.has_annotation('DEP') and (not example.reference.has_annotation('DEP', require_complete=True))):
        return example
    words = token_dict.get('ORTH', [])
    length = len(words)
    assert 0 <= position <= length
    if example.reference.has_annotation('ENT_TYPE'):
        entity = 'O'
        if position > 1 and position < length:
            ent_prev = doc_dict['entities'][position - 1]
            ent_next = doc_dict['entities'][position]
            if '-' in ent_prev and '-' in ent_next:
                ent_iob_prev, ent_type_prev = split_bilu_label(ent_prev)
                ent_iob_next, ent_type_next = split_bilu_label(ent_next)
                if ent_iob_prev in ('B', 'I') and ent_iob_next in ('I', 'L') and (ent_type_prev == ent_type_next):
                    entity = f'I-{ent_type_prev}'
        doc_dict['entities'].insert(position, entity)
    else:
        del doc_dict['entities']
    token_dict['ORTH'].insert(position, whitespace)
    token_dict['SPACY'].insert(position, False)
    if example.reference.has_annotation('TAG'):
        token_dict['TAG'].insert(position, '_SP')
    else:
        del token_dict['TAG']
    if example.reference.has_annotation('LEMMA'):
        token_dict['LEMMA'].insert(position, whitespace)
    else:
        del token_dict['LEMMA']
    if example.reference.has_annotation('POS'):
        token_dict['POS'].insert(position, 'SPACE')
    else:
        del token_dict['POS']
    if example.reference.has_annotation('MORPH'):
        token_dict['MORPH'].insert(position, '')
    else:
        del token_dict['MORPH']
    if example.reference.has_annotation('DEP', require_complete=True):
        if position == 0:
            token_dict['HEAD'].insert(position, 0)
        else:
            token_dict['HEAD'].insert(position, position - 1)
        for i in range(len(token_dict['HEAD'])):
            if token_dict['HEAD'][i] >= position:
                token_dict['HEAD'][i] += 1
        token_dict['DEP'].insert(position, 'dep')
    else:
        del token_dict['HEAD']
        del token_dict['DEP']
    if example.reference.has_annotation('SENT_START'):
        token_dict['SENT_START'].insert(position, False)
    else:
        del token_dict['SENT_START']
    raw = construct_modified_raw_text(token_dict)
    return Example.from_dict(nlp.make_doc(raw), example_dict)