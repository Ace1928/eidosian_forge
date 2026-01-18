import copy
import json
import os
import re
import warnings
from collections import UserDict
from collections.abc import Mapping, Sized
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, AddedToken]], replace_additional_special_tokens=True) -> int:
    """
        Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
        special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
        current vocabulary).

        When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the
        model so that its embedding matrix matches the tokenizer.

        In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.

        Using `add_special_tokens` will ensure your special tokens can be used in several ways:

        - Special tokens can be skipped when decoding using `skip_special_tokens = True`.
        - Special tokens are carefully handled by the tokenizer (they are never split), similar to `AddedTokens`.
        - You can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This
          makes it easy to develop model-agnostic training and fine-tuning scripts.

        When possible, special tokens are already registered for provided pretrained models (for instance
        [`BertTokenizer`] `cls_token` is already registered to be :obj*'[CLS]'* and XLM's one is also registered to be
        `'</s>'`).

        Args:
            special_tokens_dict (dictionary *str* to *str* or `tokenizers.AddedToken`):
                Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
                `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].

                Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
                assign the index of the `unk_token` to them).
            replace_additional_special_tokens (`bool`, *optional*,, defaults to `True`):
                If `True`, the existing list of additional special tokens will be replaced by the list provided in
                `special_tokens_dict`. Otherwise, `self._additional_special_tokens` is just extended. In the former
                case, the tokens will NOT be removed from the tokenizer's full vocabulary - they are only being flagged
                as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the
                `added_tokens_encoder` and `added_tokens_decoder`. This means that the previous
                `additional_special_tokens` are still added tokens, and will not be split by the model.

        Returns:
            `int`: Number of tokens added to the vocabulary.

        Examples:

        ```python
        # Let's see how to add a new classification token to GPT-2
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        model = GPT2Model.from_pretrained("openai-community/gpt2")

        special_tokens_dict = {"cls_token": "<CLS>"}

        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print("We have added", num_added_toks, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))

        assert tokenizer.cls_token == "<CLS>"
        ```"""
    if not special_tokens_dict:
        return 0
    added_tokens = []
    for key, value in special_tokens_dict.items():
        assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f'Key {key} is not a special token'
        if self.verbose:
            logger.info(f'Assigning {value} to the {key} key of the tokenizer')
        if key == 'additional_special_tokens':
            assert isinstance(value, (list, tuple)) and all((isinstance(t, (str, AddedToken)) for t in value)), f'Tokens {value} for key {key} should all be str or AddedToken instances'
            to_add = []
            for token in value:
                if isinstance(token, str):
                    token = AddedToken(token, rstrip=False, lstrip=False, normalized=False, special=True)
                if not replace_additional_special_tokens and str(token) in self.additional_special_tokens:
                    continue
                to_add.append(token)
            if replace_additional_special_tokens and len(to_add) > 0:
                setattr(self, key, list(to_add))
            else:
                self._additional_special_tokens.extend(to_add)
            added_tokens += to_add
        else:
            if not isinstance(value, (str, AddedToken)):
                raise ValueError(f'Token {value} for key {key} should be a str or an AddedToken instance')
            if isinstance(value, str):
                value = AddedToken(value, rstrip=False, lstrip=False, normalized=False, special=True)
            if isinstance(value, AddedToken):
                setattr(self, key, value)
            if value not in added_tokens:
                added_tokens.append(value)
    added_tokens = self.add_tokens(added_tokens, special_tokens=True)
    return added_tokens