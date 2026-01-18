import asyncio
import json
import os
from tempfile import TemporaryDirectory
import pytest
import tornado
def expected_http_error(error, expected_code, expected_message=None):
    """Check that the error matches the expected output error."""
    e = error.value
    if isinstance(e, tornado.web.HTTPError):
        if expected_code != e.status_code:
            return False
        if expected_message is not None and expected_message != str(e):
            return False
        return True
    elif any([isinstance(e, tornado.httpclient.HTTPClientError), isinstance(e, tornado.httpclient.HTTPError)]):
        if expected_code != e.code:
            return False
        if expected_message:
            message = json.loads(e.response.body.decode())['message']
            if expected_message != message:
                return False
        return True