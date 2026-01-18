import unittest
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testForbidden(self):
    err = exceptions.HttpError.FromResponse(_MakeResponse(403))
    self.assertIsInstance(err, exceptions.HttpError)
    self.assertIsInstance(err, exceptions.HttpForbiddenError)
    self.assertEquals(err.status_code, 403)