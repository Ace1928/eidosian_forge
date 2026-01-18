import inspect
import types
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
from ..data import SquadExample, SquadFeatures, squad_convert_examples_to_features
from ..modelcard import ModelCard
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
from .base import ArgumentHandler, ChunkPipeline, build_pipeline_init_args
def get_indices(self, enc: 'tokenizers.Encoding', s: int, e: int, sequence_index: int, align_to_words: bool) -> Tuple[int, int]:
    if align_to_words:
        try:
            start_word = enc.token_to_word(s)
            end_word = enc.token_to_word(e)
            start_index = enc.word_to_chars(start_word, sequence_index=sequence_index)[0]
            end_index = enc.word_to_chars(end_word, sequence_index=sequence_index)[1]
        except Exception:
            start_index = enc.offsets[s][0]
            end_index = enc.offsets[e][1]
    else:
        start_index = enc.offsets[s][0]
        end_index = enc.offsets[e][1]
    return (start_index, end_index)