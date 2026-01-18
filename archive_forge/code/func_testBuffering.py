from __future__ import annotations
from twisted.conch import mixin
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def testBuffering(self) -> None:
    p = TestBufferingProto()
    t = p.transport = StringTransport()
    self.assertFalse(p.scheduled)
    L = [b'foo', b'bar', b'baz', b'quux']
    p.write(b'foo')
    self.assertTrue(p.scheduled)
    self.assertFalse(p.rescheduled)
    for s in L:
        n = p.rescheduled
        p.write(s)
        self.assertEqual(p.rescheduled, n + 1)
        self.assertEqual(t.value(), b'')
    p.flush()
    self.assertEqual(t.value(), b'foo' + b''.join(L))