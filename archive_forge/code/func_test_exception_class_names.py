from boto.beanstalk.exception import simple
from tests.compat import unittest
def test_exception_class_names(self):
    error = FakeError('TooManyApplications', 400, 'foo', 'bar')
    exception = simple(error)
    self.assertEqual(exception.__class__.__name__, 'TooManyApplications')
    error = FakeError('TooManyApplicationsException', 400, 'foo', 'bar')
    exception = simple(error)
    self.assertEqual(exception.__class__.__name__, 'TooManyApplications')
    self.assertEqual(exception.message, 'bar')