from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union
import regex
from tiktoken import _tiktoken
def raise_disallowed_special_token(token: str) -> NoReturn:
    raise ValueError(f'Encountered text corresponding to disallowed special token {token!r}.\nIf you want this text to be encoded as a special token, pass it to `allowed_special`, e.g. `allowed_special={{{token!r}, ...}}`.\nIf you want this text to be encoded as normal text, disable the check for this token by passing `disallowed_special=(enc.special_tokens_set - {{{token!r}}})`.\nTo disable this check for all special tokens, pass `disallowed_special=()`.\n')