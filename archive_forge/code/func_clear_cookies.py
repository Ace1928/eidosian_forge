from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def clear_cookies(browser_context_id: typing.Optional[browser.BrowserContextID]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Clears cookies.

    :param browser_context_id: *(Optional)* Browser context to use when called on the browser endpoint.
    """
    params: T_JSON_DICT = dict()
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Storage.clearCookies', 'params': params}
    json = (yield cmd_dict)