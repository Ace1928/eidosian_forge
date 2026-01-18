from twisted.protocols import pcp
from twisted.trial import unittest
def testRegistersProducer(self):
    self.assertEqual(self.consumer.producer[0], self.producer)