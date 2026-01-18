from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def set_cookies(cookies: typing.List[network.CookieParam], browser_context_id: typing.Optional[browser.BrowserContextID]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets given cookies.

    :param cookies: Cookies to be set.
    :param browser_context_id: *(Optional)* Browser context to use when called on the browser endpoint.
    """
    params: T_JSON_DICT = dict()
    params['cookies'] = [i.to_json() for i in cookies]
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Storage.setCookies', 'params': params}
    json = (yield cmd_dict)