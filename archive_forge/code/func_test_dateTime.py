import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_dateTime(self):
    """
        Jellying L{datetime.timedelta} instances and then unjellying the result
        should produce objects which represent the values of the original
        inputs.
        """
    dtn = datetime.datetime.now()
    dtd = datetime.datetime.now() - dtn
    inputList = [dtn, dtd]
    c = jelly.jelly(inputList)
    output = jelly.unjelly(c)
    self.assertEqual(inputList, output)
    self.assertIsNot(inputList, output)