from collections import defaultdict
from typing import (
import numpy as np
from .errors import Errors
from .morphology import Morphology
from .tokens import Doc, Span, Token
from .training import Example
from .util import SimpleFrozenList, get_lang_class
@staticmethod
def score_token_attr(examples: Iterable[Example], attr: str, *, getter: Callable[[Token, str], Any]=getattr, missing_values: Set[Any]=MISSING_VALUES, **cfg) -> Dict[str, Any]:
    """Returns an accuracy score for a token-level attribute.

        examples (Iterable[Example]): Examples to score
        attr (str): The attribute to score.
        getter (Callable[[Token, str], Any]): Defaults to getattr. If provided,
            getter(token, attr) should return the value of the attribute for an
            individual token.
        missing_values (Set[Any]): Attribute values to treat as missing annotation
            in the reference annotation.
        RETURNS (Dict[str, Any]): A dictionary containing the accuracy score
            under the key attr_acc.

        DOCS: https://spacy.io/api/scorer#score_token_attr
        """
    tag_score = PRFScore()
    for example in examples:
        gold_doc = example.reference
        pred_doc = example.predicted
        align = example.alignment
        gold_tags = set()
        missing_indices = set()
        for gold_i, token in enumerate(gold_doc):
            value = getter(token, attr)
            if value not in missing_values:
                gold_tags.add((gold_i, getter(token, attr)))
            else:
                missing_indices.add(gold_i)
        pred_tags = set()
        for token in pred_doc:
            if token.orth_.isspace():
                continue
            if align.x2y.lengths[token.i] == 1:
                gold_i = align.x2y[token.i][0]
                if gold_i not in missing_indices:
                    pred_tags.add((gold_i, getter(token, attr)))
        tag_score.score_set(pred_tags, gold_tags)
    score_key = f'{attr}_acc'
    if len(tag_score) == 0:
        return {score_key: None}
    else:
        return {score_key: tag_score.fscore}