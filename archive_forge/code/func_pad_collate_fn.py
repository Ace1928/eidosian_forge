import collections
import csv
import importlib
import json
import os
import pickle
import sys
import traceback
import types
import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from contextlib import contextmanager
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from ..dynamic_module_utils import custom_object_save
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..modelcard import ModelCard
from ..models.auto.configuration_auto import AutoConfig
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
def pad_collate_fn(tokenizer, feature_extractor):
    t_padding_side = None
    f_padding_side = None
    if tokenizer is None and feature_extractor is None:
        raise ValueError('Pipeline without tokenizer or feature_extractor cannot do batching')
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            raise ValueError('Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with `pipe.tokenizer.pad_token_id = model.config.eos_token_id`.')
        else:
            t_padding_value = tokenizer.pad_token_id
            t_padding_side = tokenizer.padding_side
    if feature_extractor is not None:
        f_padding_value = getattr(feature_extractor, 'padding_value', None)
        f_padding_side = getattr(feature_extractor, 'padding_side', None)
    if t_padding_side is not None and f_padding_side is not None and (t_padding_side != f_padding_side):
        raise ValueError(f"The feature extractor, and tokenizer don't agree on padding side {t_padding_side} != {f_padding_side}")
    padding_side = 'right'
    if t_padding_side is not None:
        padding_side = t_padding_side
    if f_padding_side is not None:
        padding_side = f_padding_side

    def inner(items):
        keys = set(items[0].keys())
        for item in items:
            if set(item.keys()) != keys:
                raise ValueError(f'The elements of the batch contain different keys. Cannot batch them ({set(item.keys())} != {keys})')
        padded = {}
        for key in keys:
            if key in {'input_ids'}:
                if tokenizer is None and feature_extractor is not None:
                    _padding_value = f_padding_value
                else:
                    _padding_value = t_padding_value
            elif key in {'input_values', 'pixel_values', 'input_features'}:
                _padding_value = f_padding_value
            elif key in {'p_mask', 'special_tokens_mask'}:
                _padding_value = 1
            elif key in {'attention_mask', 'token_type_ids'}:
                _padding_value = 0
            else:
                _padding_value = 0
            padded[key] = _pad(items, key, _padding_value, padding_side)
        return padded
    return inner