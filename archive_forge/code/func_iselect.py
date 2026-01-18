from __future__ import annotations
from .__meta__ import __version__, __version_info__  # noqa: F401
from . import css_parser as cp
from . import css_match as cm
from . import css_types as ct
from .util import DEBUG, SelectorSyntaxError  # noqa: F401
import bs4  # type: ignore[import]
from typing import Any, Iterator, Iterable
def iselect(select: str, tag: bs4.Tag, namespaces: dict[str, str] | None=None, limit: int=0, flags: int=0, *, custom: dict[str, str] | None=None, **kwargs: Any) -> Iterator[bs4.Tag]:
    """Iterate the specified tags."""
    yield from compile(select, namespaces, flags, **kwargs).iselect(tag, limit)