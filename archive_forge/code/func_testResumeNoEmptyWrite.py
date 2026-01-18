from twisted.protocols import pcp
from twisted.trial import unittest
def testResumeNoEmptyWrite(self):
    self.producer.pauseProducing()
    self.producer.resumeProducing()
    self.assertEqual(len(self.consumer._writes), 0, 'Resume triggered an empty write.')