import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_from_function_and_args(self):
    ctx = context.RequestContext(user='user1')
    arg = []
    kw = dict(c=ctx, s='s')
    fn = context.get_context_from_function_and_args
    ctx1 = context.get_context_from_function_and_args(fn, arg, kw)
    self.assertIs(ctx1, ctx)