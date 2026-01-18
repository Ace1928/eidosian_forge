from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
def create_browser_context(dispose_on_detach: typing.Optional[bool]=None, proxy_server: typing.Optional[str]=None, proxy_bypass_list: typing.Optional[str]=None, origins_with_universal_network_access: typing.Optional[typing.List[str]]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, browser.BrowserContextID]:
    """
    Creates a new empty BrowserContext. Similar to an incognito profile but you can have more than
    one.

    :param dispose_on_detach: **(EXPERIMENTAL)** *(Optional)* If specified, disposes this context when debugging session disconnects.
    :param proxy_server: **(EXPERIMENTAL)** *(Optional)* Proxy server, similar to the one passed to --proxy-server
    :param proxy_bypass_list: **(EXPERIMENTAL)** *(Optional)* Proxy bypass list, similar to the one passed to --proxy-bypass-list
    :param origins_with_universal_network_access: **(EXPERIMENTAL)** *(Optional)* An optional list of origins to grant unlimited cross-origin access to. Parts of the URL other than those constituting origin are ignored.
    :returns: The id of the context created.
    """
    params: T_JSON_DICT = dict()
    if dispose_on_detach is not None:
        params['disposeOnDetach'] = dispose_on_detach
    if proxy_server is not None:
        params['proxyServer'] = proxy_server
    if proxy_bypass_list is not None:
        params['proxyBypassList'] = proxy_bypass_list
    if origins_with_universal_network_access is not None:
        params['originsWithUniversalNetworkAccess'] = [i for i in origins_with_universal_network_access]
    cmd_dict: T_JSON_DICT = {'method': 'Target.createBrowserContext', 'params': params}
    json = (yield cmd_dict)
    return browser.BrowserContextID.from_json(json['browserContextId'])