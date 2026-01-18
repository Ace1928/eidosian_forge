import socket
import io
import re
import email.utils
import email.message
import email.generator
import base64
import hmac
import copy
import datetime
import sys
from email.base64mime import body_encode as encode_base64
def ehlo_or_helo_if_needed(self):
    """Call self.ehlo() and/or self.helo() if needed.

        If there has been no previous EHLO or HELO command this session, this
        method tries ESMTP EHLO first.

        This method may raise the following exceptions:

         SMTPHeloError            The server didn't reply properly to
                                  the helo greeting.
        """
    if self.helo_resp is None and self.ehlo_resp is None:
        if not 200 <= self.ehlo()[0] <= 299:
            code, resp = self.helo()
            if not 200 <= code <= 299:
                raise SMTPHeloError(code, resp)