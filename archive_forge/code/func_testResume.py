from twisted.protocols import pcp
from twisted.trial import unittest
def testResume(self):
    self.producer.pauseProducing()
    self.producer.resumeProducing()
    self.producer.write('yakkity yak')
    self.assertEqual(self.consumer.getvalue(), 'yakkity yak')