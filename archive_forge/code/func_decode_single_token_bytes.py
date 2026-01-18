from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union
import regex
from tiktoken import _tiktoken
def decode_single_token_bytes(self, token: int) -> bytes:
    """Decodes a token into bytes.

        NOTE: this will decode all special tokens.

        Raises `KeyError` if the token is not in the vocabulary.

        ```
        >>> enc.decode_single_token_bytes(31373)
        b'hello'
        ```
        """
    return self._core_bpe.decode_single_token_bytes(token)