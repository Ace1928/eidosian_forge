import json
import re
import socket
import time
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from os import path
from queue import PriorityQueue, Queue
from threading import Thread
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union, cast
from urllib.parse import unquote, urlparse, urlunparse
from docutils import nodes
from requests import Response
from requests.exceptions import ConnectionError, HTTPError, TooManyRedirects
from sphinx.application import Sphinx
from sphinx.builders.dummy import DummyBuilder
from sphinx.config import Config
from sphinx.environment import BuildEnvironment
from sphinx.locale import __
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import encode_uri, logging, requests
from sphinx.util.console import darkgray, darkgreen, purple, red, turquoise  # type: ignore
from sphinx.util.nodes import get_node_line
def check_uri() -> Tuple[str, str, int]:
    if '#' in uri:
        req_url, anchor = uri.split('#', 1)
        for rex in self.anchors_ignore:
            if rex.match(anchor):
                anchor = None
                break
    else:
        req_url = uri
        anchor = None
    try:
        req_url.encode('ascii')
    except UnicodeError:
        req_url = encode_uri(req_url)
    for pattern, auth_info in self.auth:
        if pattern.match(uri):
            break
    else:
        auth_info = None
    kwargs['headers'] = get_request_headers()
    try:
        if anchor and self.config.linkcheck_anchors:
            response = requests.get(req_url, stream=True, config=self.config, auth=auth_info, **kwargs)
            response.raise_for_status()
            found = check_anchor(response, unquote(anchor))
            if not found:
                raise Exception(__("Anchor '%s' not found") % anchor)
        else:
            try:
                response = requests.head(req_url, allow_redirects=True, config=self.config, auth=auth_info, **kwargs)
                response.raise_for_status()
            except (ConnectionError, HTTPError, TooManyRedirects) as err:
                if isinstance(err, HTTPError) and err.response.status_code == 429:
                    raise
                response = requests.get(req_url, stream=True, config=self.config, auth=auth_info, **kwargs)
                response.raise_for_status()
    except HTTPError as err:
        if err.response.status_code == 401:
            return ('working', ' - unauthorized', 0)
        elif err.response.status_code == 429:
            next_check = self.limit_rate(err.response)
            if next_check is not None:
                self.wqueue.put(CheckRequest(next_check, hyperlink), False)
                return ('rate-limited', '', 0)
            return ('broken', str(err), 0)
        elif err.response.status_code == 503:
            return ('ignored', str(err), 0)
        else:
            return ('broken', str(err), 0)
    except Exception as err:
        return ('broken', str(err), 0)
    else:
        netloc = urlparse(req_url).netloc
        try:
            del self.rate_limits[netloc]
        except KeyError:
            pass
    if response.url.rstrip('/') == req_url.rstrip('/'):
        return ('working', '', 0)
    else:
        new_url = response.url
        if anchor:
            new_url += '#' + anchor
        if allowed_redirect(req_url, new_url):
            return ('working', '', 0)
        elif response.history:
            code = response.history[-1].status_code
            return ('redirected', new_url, code)
        else:
            return ('redirected', new_url, 0)