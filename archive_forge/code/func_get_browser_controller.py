from __future__ import annotations
import logging # isort:skip
import webbrowser
from os.path import abspath
from typing import Literal, Protocol, cast
from ..settings import settings
def get_browser_controller(browser: str | None=None) -> BrowserLike:
    """ Return a browser controller.

    Args:
        browser (str or None) : browser name, or ``None`` (default: ``None``)
            If passed the string ``'none'``, a dummy web browser controller
            is returned.

            Otherwise, use the value to select an appropriate controller using
            the :doc:`webbrowser <python:library/webbrowser>` standard library
            module. If the value is ``None``, a system default is used.

    Returns:
        controller : a web browser controller

    """
    browser = settings.browser(browser)
    if browser is None:
        controller = cast(BrowserLike, webbrowser)
    elif browser == 'none':
        controller = DummyWebBrowser()
    else:
        controller = webbrowser.get(browser)
    return controller