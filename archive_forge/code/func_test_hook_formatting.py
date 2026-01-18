import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
def test_hook_formatting(self):
    hooks = ['<pecan.hooks.RequestViewerHook object at 0x103a5f910>']
    viewer = RequestViewerHook()
    formatted = viewer.format_hooks(hooks)
    assert formatted == ['RequestViewerHook']