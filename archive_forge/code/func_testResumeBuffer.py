from twisted.protocols import pcp
from twisted.trial import unittest
def testResumeBuffer(self):
    self.producer.pauseProducing()
    self.producer.write('buffer this')
    self.producer.resumeProducing()
    self.assertEqual(self.consumer.getvalue(), 'buffer this')