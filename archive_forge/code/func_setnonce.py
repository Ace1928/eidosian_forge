from logging import getLogger
from suds import *
from suds.sudsobject import Object
from suds.sax.element import Element
from suds.sax.date import DateTime, UtcTimezone
from datetime import datetime, timedelta
def setnonce(self, text=None):
    """
        Set I{nonce} which is an arbitrary set of bytes to prevent replay
        attacks.
        @param text: The nonce text value.
            Generated when I{None}.
        @type text: str
        """
    if text is None:
        s = []
        s.append(self.username)
        s.append(self.password)
        s.append(Token.sysdate())
        try:
            m = md5(usedforsecurity=False)
        except (AttributeError, TypeError):
            m = md5()
        m.update(':'.join(s).encode('utf-8'))
        self.nonce = m.hexdigest()
    else:
        self.nonce = text