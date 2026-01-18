from __future__ import absolute_import, division, print_function
import logging
from passlib import hash, exc
from passlib.utils.compat import u
from .utils import UserHandlerMixin, HandlerCase, repeat_string
from .test_handlers import UPASS_TABLE
def test_calc_digest_spoiler(self):
    """
        _calc_checksum() -- spoil oversize passwords during verify

        for details, see 'spoil_digest' flag instead that function.
        this helps cisco_pix/cisco_asa implement their policy of
        ``.truncate_verify_reject=True``.
        """

    def calc(secret, for_hash=False):
        return self.handler(use_defaults=for_hash)._calc_checksum(secret)
    short_secret = repeat_string('1234', self.handler.truncate_size)
    short_hash = calc(short_secret)
    long_secret = short_secret + 'X'
    long_hash = calc(long_secret)
    self.assertNotEqual(long_hash, short_hash)
    alt_long_secret = short_secret + 'Y'
    alt_long_hash = calc(alt_long_secret)
    self.assertNotEqual(alt_long_hash, short_hash)
    self.assertNotEqual(alt_long_hash, long_hash)
    calc(short_secret, for_hash=True)
    self.assertRaises(exc.PasswordSizeError, calc, long_secret, for_hash=True)
    self.assertRaises(exc.PasswordSizeError, calc, alt_long_secret, for_hash=True)