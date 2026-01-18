import dataclasses
from typing import Collection
from werkzeug.datastructures import Headers
from werkzeug import http
from tensorboard.util import tb_logging
def start_response_proxy(status, headers, exc_info=None):
    self._validate_headers(headers)
    return start_response(status, headers, exc_info)