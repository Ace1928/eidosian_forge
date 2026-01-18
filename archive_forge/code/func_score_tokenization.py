from collections import defaultdict
from typing import (
import numpy as np
from .errors import Errors
from .morphology import Morphology
from .tokens import Doc, Span, Token
from .training import Example
from .util import SimpleFrozenList, get_lang_class
@staticmethod
def score_tokenization(examples: Iterable[Example], **cfg) -> Dict[str, Any]:
    """Returns accuracy and PRF scores for tokenization.
        * token_acc: # correct tokens / # gold tokens
        * token_p/r/f: PRF for token character spans

        examples (Iterable[Example]): Examples to score
        RETURNS (Dict[str, Any]): A dictionary containing the scores
            token_acc/p/r/f.

        DOCS: https://spacy.io/api/scorer#score_tokenization
        """
    acc_score = PRFScore()
    prf_score = PRFScore()
    for example in examples:
        gold_doc = example.reference
        pred_doc = example.predicted
        if gold_doc.has_unknown_spaces:
            continue
        align = example.alignment
        gold_spans = set()
        pred_spans = set()
        for token in gold_doc:
            if token.orth_.isspace():
                continue
            gold_spans.add((token.idx, token.idx + len(token)))
        for token in pred_doc:
            if token.orth_.isspace():
                continue
            pred_spans.add((token.idx, token.idx + len(token)))
            if align.x2y.lengths[token.i] != 1:
                acc_score.fp += 1
            else:
                acc_score.tp += 1
        prf_score.score_set(pred_spans, gold_spans)
    if len(acc_score) > 0:
        return {'token_acc': acc_score.precision, 'token_p': prf_score.precision, 'token_r': prf_score.recall, 'token_f': prf_score.fscore}
    else:
        return {'token_acc': None, 'token_p': None, 'token_r': None, 'token_f': None}