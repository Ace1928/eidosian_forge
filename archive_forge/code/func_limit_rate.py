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
def limit_rate(self, response: Response) -> Optional[float]:
    next_check = None
    retry_after = response.headers.get('Retry-After')
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            try:
                until = parsedate_to_datetime(retry_after)
            except (TypeError, ValueError):
                pass
            else:
                next_check = datetime.timestamp(until)
                delay = (until - datetime.now(timezone.utc)).total_seconds()
        else:
            next_check = time.time() + delay
    netloc = urlparse(response.url).netloc
    if next_check is None:
        max_delay = self.config.linkcheck_rate_limit_timeout
        try:
            rate_limit = self.rate_limits[netloc]
        except KeyError:
            delay = DEFAULT_DELAY
        else:
            last_wait_time = rate_limit.delay
            delay = 2.0 * last_wait_time
            if delay > max_delay and last_wait_time < max_delay:
                delay = max_delay
        if delay > max_delay:
            return None
        next_check = time.time() + delay
    self.rate_limits[netloc] = RateLimit(delay, next_check)
    return next_check