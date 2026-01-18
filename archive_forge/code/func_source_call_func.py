import json
import os
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import regex as re
from ....file_utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available
from ....tokenization_utils import AddedToken, PreTrainedTokenizer
from ....tokenization_utils_base import ENCODE_KWARGS_DOCSTRING, BatchEncoding, TextInput, TruncationStrategy
from ....utils import logging
def source_call_func(self, table: Union['pd.DataFrame', List['pd.DataFrame']], query: Optional[Union[TextInput, List[TextInput]]]=None, answer: Union[str, List[str]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
    valid_table = False
    valid_query = False
    if isinstance(table, pd.DataFrame):
        valid_table = True
    elif isinstance(table, (list, tuple)) and isinstance(table[0], pd.DataFrame):
        valid_table = True
    if query is None or isinstance(query, str):
        valid_query = True
    elif isinstance(query, (list, tuple)):
        if len(query) == 0 or isinstance(query[0], str):
            valid_query = True
    if not valid_table:
        raise ValueError('table input must of type `pd.DataFrame` (single example), `List[pd.DataFrame]` (batch of examples). ')
    if not valid_query:
        raise ValueError('query input must of type `str` (single example), `List[str]` (batch of examples). ')
    is_batched = isinstance(table, (list, tuple)) or isinstance(query, (list, tuple))
    if is_batched:
        return self.batch_encode_plus(table=table, query=query, answer=answer, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)
    else:
        return self.encode_plus(table=table, query=query, answer=answer, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)