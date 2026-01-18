import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
def test_mixed_args(self):
    self.app.get('/mixed/foo/bar/spam/eggs')
    assert self.args[0] == self.root.mixed
    assert isinstance(self.args[1], inspect.Arguments)
    assert len(self.args[1].args) == 4
    assert isinstance(self.args[1].args[0], Request)
    assert isinstance(self.args[1].args[1], Response)
    assert self.args[1].args[2:] == ['foo', 'bar']
    assert self.args[1].varargs == ['spam', 'eggs']