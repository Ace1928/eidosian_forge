import re
import sys
import traceback
from typing import NoReturn
import pytest
from .._util import (
def test_LocalProtocolError() -> None:
    try:
        raise LocalProtocolError('foo')
    except LocalProtocolError as e:
        assert str(e) == 'foo'
        assert e.error_status_hint == 400
    try:
        raise LocalProtocolError('foo', error_status_hint=418)
    except LocalProtocolError as e:
        assert str(e) == 'foo'
        assert e.error_status_hint == 418

    def thunk() -> NoReturn:
        raise LocalProtocolError('a', error_status_hint=420)
    try:
        try:
            thunk()
        except LocalProtocolError as exc1:
            orig_traceback = ''.join(traceback.format_tb(sys.exc_info()[2]))
            exc1._reraise_as_remote_protocol_error()
    except RemoteProtocolError as exc2:
        assert type(exc2) is RemoteProtocolError
        assert exc2.args == ('a',)
        assert exc2.error_status_hint == 420
        new_traceback = ''.join(traceback.format_tb(sys.exc_info()[2]))
        assert new_traceback.endswith(orig_traceback)