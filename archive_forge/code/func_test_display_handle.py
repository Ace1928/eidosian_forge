import json
import os
import warnings
from unittest import mock
import pytest
from IPython import display
from IPython.core.getipython import get_ipython
from IPython.utils.io import capture_output
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython import paths as ipath
from IPython.testing.tools import AssertNotPrints
import IPython.testing.decorators as dec
def test_display_handle():
    ip = get_ipython()
    handle = display.DisplayHandle()
    assert isinstance(handle.display_id, str)
    handle = display.DisplayHandle('my-id')
    assert handle.display_id == 'my-id'
    with mock.patch.object(ip.display_pub, 'publish') as pub:
        handle.display('x')
        handle.update('y')
    args, kwargs = pub.call_args_list[0]
    assert args == ()
    assert kwargs == {'data': {'text/plain': repr('x')}, 'metadata': {}, 'transient': {'display_id': handle.display_id}}
    args, kwargs = pub.call_args_list[1]
    assert args == ()
    assert kwargs == {'data': {'text/plain': repr('y')}, 'metadata': {}, 'transient': {'display_id': handle.display_id}, 'update': True}