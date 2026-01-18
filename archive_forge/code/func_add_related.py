import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def add_related(self, *args, **kw):
    self._add_multipart('related', *args, _disp='inline', **kw)