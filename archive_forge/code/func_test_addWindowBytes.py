from __future__ import annotations
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.trial.unittest import TestCase
def test_addWindowBytes(self) -> None:
    """
        Test that addWindowBytes adds bytes to the window and resumes writing
        if it was paused.
        """
    cb = [False]

    def stubStartWriting() -> None:
        cb[0] = True
    self.channel.startWriting = stubStartWriting
    self.channel.write(b'test')
    self.channel.writeExtended(1, b'test')
    self.channel.addWindowBytes(50)
    self.assertEqual(self.channel.remoteWindowLeft, 50 - 4 - 4)
    self.assertTrue(self.channel.areWriting)
    self.assertTrue(cb[0])
    self.assertEqual(self.channel.buf, b'')
    self.assertEqual(self.conn.data[self.channel], [b'test'])
    self.assertEqual(self.channel.extBuf, [])
    self.assertEqual(self.conn.extData[self.channel], [(1, b'test')])
    cb[0] = False
    self.channel.addWindowBytes(20)
    self.assertFalse(cb[0])
    self.channel.write(b'a' * 80)
    self.channel.loseConnection()
    self.channel.addWindowBytes(20)
    self.assertFalse(cb[0])