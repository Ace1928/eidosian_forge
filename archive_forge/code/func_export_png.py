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
def export_png(obj: UIElement | Document, *, filename: PathLike | None=None, width: int | None=None, height: int | None=None, scale_factor: float=1, webdriver: WebDriver | None=None, timeout: int=5, state: State | None=None) -> str:
    """ Export the ``UIElement`` object or document as a PNG.

    If the filename is not given, it is derived from the script name (e.g.
    ``/foo/myplot.py`` will create ``/foo/myplot.png``)

    Args:
        obj (UIElement or Document) : a Layout (Row/Column), Plot or Widget
            object or Document to export.

        filename (PathLike, e.g. str, Path, optional) : filename to save document under (default: None)
            If None, infer from the filename.

        width (int) : the desired width of the exported layout obj only if
            it's a Plot instance. Otherwise the width kwarg is ignored.

        height (int) : the desired height of the exported layout obj only if
            it's a Plot instance. Otherwise the height kwarg is ignored.

        scale_factor (float, optional) : A factor to scale the output PNG by,
            providing a higher resolution while maintaining element relative
            scales.

        webdriver (selenium.webdriver) : a selenium webdriver instance to use
            to export the image.

        timeout (int) : the maximum amount of time (in seconds) to wait for
            Bokeh to initialize (default: 5) (Added in 1.1.1).

        state (State, optional) :
            A :class:`State` object. If None, then the current default
            implicit state is used. (default: None).

    Returns:
        filename (str) : the filename where the static file is saved.

    If you would like to access an Image object directly, rather than save a
    file to disk, use the lower-level :func:`~bokeh.io.export.get_screenshot_as_png`
    function.

    .. warning::
        Responsive sizing_modes may generate layouts with unexpected size and
        aspect ratios. It is recommended to use the default ``fixed`` sizing mode.

    """
    image = get_screenshot_as_png(obj, width=width, height=height, scale_factor=scale_factor, driver=webdriver, timeout=timeout, state=state)
    if filename is None:
        filename = default_filename('png')
    if image.width == 0 or image.height == 0:
        raise ValueError('unable to save an empty image')
    filename = os.fspath(filename)
    image.save(filename)
    return abspath(expanduser(filename))