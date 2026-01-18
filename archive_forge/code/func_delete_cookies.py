from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
def delete_cookies(name: str, url: typing.Optional[str]=None, domain: typing.Optional[str]=None, path: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Deletes browser cookies with matching name and url or domain/path pair.

    :param name: Name of the cookies to remove.
    :param url: *(Optional)* If specified, deletes all the cookies with the given name where domain and path match provided URL.
    :param domain: *(Optional)* If specified, deletes only cookies with the exact domain.
    :param path: *(Optional)* If specified, deletes only cookies with the exact path.
    """
    params: T_JSON_DICT = dict()
    params['name'] = name
    if url is not None:
        params['url'] = url
    if domain is not None:
        params['domain'] = domain
    if path is not None:
        params['path'] = path
    cmd_dict: T_JSON_DICT = {'method': 'Network.deleteCookies', 'params': params}
    json = (yield cmd_dict)