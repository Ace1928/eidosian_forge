from __future__ import annotations
from typing import Type
from twisted.internet.defer import Deferred, succeed
from twisted.trial.unittest import SynchronousTestCase
from twisted.web import server
from twisted.web._flatten import flattenString
from twisted.web.error import FlattenerError
from twisted.web.http import Request
from twisted.web.resource import IResource
from twisted.web.template import Flattenable
from .requesthelper import DummyRequest
def assertFlatteningRaises(self, root: Flattenable, exn: Type[Exception]) -> None:
    """
        Assert flattening a root element raises a particular exception.
        """
    failure = self.failureResultOf(self.assertFlattensTo(root, b''), FlattenerError)
    self.assertIsInstance(failure.value._exception, exn)