from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union
import regex
from tiktoken import _tiktoken
def decode_with_offsets(self, tokens: list[int]) -> tuple[str, list[int]]:
    """Decodes a list of tokens into a string and a list of offsets.

        Each offset is the index into text corresponding to the start of each token.
        If UTF-8 character boundaries do not line up with token boundaries, the offset is the index
        of the first character that contains bytes from the token.

        This will currently raise if given tokens that decode to invalid UTF-8; this behaviour may
        change in the future to be more permissive.

        >>> enc.decode_with_offsets([31373, 995])
        ('hello world', [0, 5])
        """
    token_bytes = self.decode_tokens_bytes(tokens)
    text_len = 0
    offsets = []
    for token in token_bytes:
        offsets.append(max(0, text_len - (128 <= token[0] < 192)))
        text_len += sum((1 for c in token if not 128 <= c < 192))
    text = b''.join(token_bytes).decode('utf-8', errors='strict')
    return (text, offsets)