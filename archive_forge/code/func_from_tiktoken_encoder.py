from __future__ import annotations
import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
from langchain_core.documents import BaseDocumentTransformer, Document
@classmethod
def from_tiktoken_encoder(cls: Type[TS], encoding_name: str='gpt2', model_name: Optional[str]=None, allowed_special: Union[Literal['all'], AbstractSet[str]]=set(), disallowed_special: Union[Literal['all'], Collection[str]]='all', **kwargs: Any) -> TS:
    """Text splitter that uses tiktoken encoder to count length."""
    try:
        import tiktoken
    except ImportError:
        raise ImportError('Could not import tiktoken python package. This is needed in order to calculate max_tokens_for_prompt. Please install it with `pip install tiktoken`.')
    if model_name is not None:
        enc = tiktoken.encoding_for_model(model_name)
    else:
        enc = tiktoken.get_encoding(encoding_name)

    def _tiktoken_encoder(text: str) -> int:
        return len(enc.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special))
    if issubclass(cls, TokenTextSplitter):
        extra_kwargs = {'encoding_name': encoding_name, 'model_name': model_name, 'allowed_special': allowed_special, 'disallowed_special': disallowed_special}
        kwargs = {**kwargs, **extra_kwargs}
    return cls(length_function=_tiktoken_encoder, **kwargs)