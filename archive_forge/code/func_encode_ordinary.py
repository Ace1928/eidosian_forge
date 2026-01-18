from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union
import regex
from tiktoken import _tiktoken
def encode_ordinary(self, text: str) -> list[int]:
    """Encodes a string into tokens, ignoring special tokens.

        This is equivalent to `encode(text, disallowed_special=())` (but slightly faster).

        ```
        >>> enc.encode_ordinary("hello world")
        [31373, 995]
        """
    try:
        return self._core_bpe.encode_ordinary(text)
    except UnicodeEncodeError:
        text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'replace')
        return self._core_bpe.encode_ordinary(text)