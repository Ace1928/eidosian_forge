from __future__ import annotations
import email.message
import email.parser
from io import BytesIO, StringIO
from typing import IO, AnyStr, Callable
from twisted.mail import bounce
from twisted.trial import unittest
def test_bounceMessageBytes(self) -> None:
    """
        L{twisted.mail.bounce.generateBounce} can accept L{bytes}.
        """
    fromAddress, to, s = bounce.generateBounce(BytesIO(b'From: Moshe Zadka <moshez@example.com>\nTo: nonexistent@example.org\nSubject: test\n\n'), b'moshez@example.com', b'nonexistent@example.org')
    self.assertEqual(fromAddress, b'')
    self.assertEqual(to, b'moshez@example.com')
    emailParser = email.parser.Parser()
    mess = emailParser.parse(StringIO(s.decode('utf-8')))
    self.assertEqual(mess['To'], 'moshez@example.com')
    self.assertEqual(mess['From'], 'postmaster@example.org')
    self.assertEqual(mess['subject'], 'Returned Mail: see transcript for details')