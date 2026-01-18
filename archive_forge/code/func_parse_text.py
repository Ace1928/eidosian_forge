import collections
import datetime
import enum
import itertools
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Text, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
from ...utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available, logging
def parse_text(text):
    """
    Extracts longest number and date spans.

    Args:
      text: text to annotate

    Returns:
      List of longest numeric value spans.
    """
    span_dict = collections.defaultdict(list)
    for match in _NUMBER_PATTERN.finditer(text):
        span_text = text[match.start():match.end()]
        number = _parse_number(span_text)
        if number is not None:
            span_dict[match.span()].append(_get_numeric_value_from_float(number))
    for begin_index, end_index in get_all_spans(text, max_ngram_length=1):
        if (begin_index, end_index) in span_dict:
            continue
        span_text = text[begin_index:end_index]
        number = _parse_number(span_text)
        if number is not None:
            span_dict[begin_index, end_index].append(_get_numeric_value_from_float(number))
        for number, word in enumerate(_NUMBER_WORDS):
            if span_text == word:
                span_dict[begin_index, end_index].append(_get_numeric_value_from_float(float(number)))
                break
        for number, word in enumerate(_ORDINAL_WORDS):
            if span_text == word:
                span_dict[begin_index, end_index].append(_get_numeric_value_from_float(float(number)))
                break
    for begin_index, end_index in get_all_spans(text, max_ngram_length=_MAX_DATE_NGRAM_SIZE):
        span_text = text[begin_index:end_index]
        date = _parse_date(span_text)
        if date is not None:
            span_dict[begin_index, end_index].append(date)
    spans = sorted(span_dict.items(), key=lambda span_value: _get_span_length_key(span_value[0]), reverse=True)
    selected_spans = []
    for span, value in spans:
        for selected_span, _ in selected_spans:
            if selected_span[0] <= span[0] and span[1] <= selected_span[1]:
                break
        else:
            selected_spans.append((span, value))
    selected_spans.sort(key=lambda span_value: span_value[0][0])
    numeric_value_spans = []
    for span, values in selected_spans:
        numeric_value_spans.append(NumericValueSpan(begin_index=span[0], end_index=span[1], values=values))
    return numeric_value_spans