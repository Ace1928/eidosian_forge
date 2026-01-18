import unittest
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testNotFound(self):
    err = exceptions.HttpError.FromResponse(_MakeResponse(404))
    self.assertIsInstance(err, exceptions.HttpError)
    self.assertIsInstance(err, exceptions.HttpNotFoundError)
    self.assertEquals(err.status_code, 404)