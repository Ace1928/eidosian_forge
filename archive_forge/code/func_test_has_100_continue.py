import pytest
from .._events import Request
from .._headers import (
from .._util import LocalProtocolError
def test_has_100_continue() -> None:
    assert has_expect_100_continue(Request(method='GET', target='/', headers=[('Host', 'example.com'), ('Expect', '100-continue')]))
    assert not has_expect_100_continue(Request(method='GET', target='/', headers=[('Host', 'example.com')]))
    assert has_expect_100_continue(Request(method='GET', target='/', headers=[('Host', 'example.com'), ('Expect', '100-Continue')]))
    assert not has_expect_100_continue(Request(method='GET', target='/', headers=[('Host', 'example.com'), ('Expect', '100-continue')], http_version='1.0'))