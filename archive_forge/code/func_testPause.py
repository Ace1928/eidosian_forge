from twisted.protocols import pcp
from twisted.trial import unittest
def testPause(self):
    self.producer.pauseProducing()
    self.producer.write('yakkity yak')
    self.assertFalse(self.consumer.getvalue(), 'Paused producer should not have sent data.')