from __future__ import annotations
import pathlib
from typing import TYPE_CHECKING, Literal
import param
from ...config import config
from ...io.resources import JS_URLS
from ..base import BasicTemplate
def resolve_resources(self, cdn: bool | Literal['auto']='auto', extras: dict[str, dict[str, str]] | None=None) -> ResourcesType:
    resources = super().resolve_resources(cdn=cdn, extras=extras)
    del_theme = 'dark' if self._design.theme._name == 'default' else 'light'
    del resources['css'][f'golden-theme-{del_theme}']
    return resources