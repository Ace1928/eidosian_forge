from __future__ import annotations
import logging # isort:skip
import io
import os
from contextlib import contextmanager
from os.path import abspath, expanduser, splitext
from tempfile import mkstemp
from typing import (
from ..core.types import PathLike
from ..document import Document
from ..embed import file_html
from ..resources import INLINE, Resources
from ..themes import Theme
from ..util.warnings import warn
from .state import State, curstate
from .util import default_filename
def get_svg(obj: UIElement | Document, *, driver: WebDriver | None=None, timeout: int=5, resources: Resources=INLINE, width: int | None=None, height: int | None=None, state: State | None=None) -> list[str]:
    from .webdriver import webdriver_control
    with _tmp_html() as tmp:
        theme = (state or curstate()).document.theme
        html = get_layout_html(obj, resources=resources, width=width, height=height, theme=theme)
        with open(tmp.path, mode='w', encoding='utf-8') as file:
            file.write(html)
        web_driver = driver if driver is not None else webdriver_control.get()
        web_driver.get(f'file://{tmp.path}')
        wait_until_render_complete(web_driver, timeout)
        svgs = cast(list[str], web_driver.execute_script(_SVG_SCRIPT))
    return svgs