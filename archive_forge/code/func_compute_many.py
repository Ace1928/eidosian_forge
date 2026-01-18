import re
from abc import ABC, abstractmethod
from collections import Counter
import functools
import datetime
from typing import Union, List, Optional, Tuple, Set, Any, Dict
import torch
from parlai.core.message import Message
from parlai.utils.misc import warn_once
from parlai.utils.typing import TScalar, TVector
@staticmethod
def compute_many(guess: str, answers: List[str]) -> Tuple[Optional['RougeMetric'], Optional['RougeMetric'], Optional['RougeMetric']]:
    """
        Compute ROUGE score between guess and *any* answer.

        Done with compute_many due to increased efficiency.

        :return: (rouge-1, rouge-2, rouge-L)
        """
    global rouge
    if rouge is None:
        return (None, None, None)
    if RougeMetric._evaluator is None:
        RougeMetric._evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)
    try:
        scores = [RougeMetric._evaluator.get_scores(normalize_answer(guess), normalize_answer(a)) for a in answers]
    except LookupError:
        warn_once('ROUGE requires nltk punkt tokenizer. Please run `python -c "import nltk; nltk.download(\'punkt\')`')
        return (None, None, None)
    scores_rouge1 = max((score['rouge-1']['r'] for score in scores))
    scores_rouge2 = max((score['rouge-2']['r'] for score in scores))
    scores_rougeL = max((score['rouge-l']['r'] for score in scores))
    return (RougeMetric(scores_rouge1), RougeMetric(scores_rouge2), RougeMetric(scores_rougeL))