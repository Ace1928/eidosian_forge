import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def set_charset(self, charset):
    """Set the charset of the payload to a given character set.

        charset can be a Charset instance, a string naming a character set, or
        None.  If it is a string it will be converted to a Charset instance.
        If charset is None, the charset parameter will be removed from the
        Content-Type field.  Anything else will generate a TypeError.

        The message will be assumed to be of type text/* encoded with
        charset.input_charset.  It will be converted to charset.output_charset
        and encoded properly, if needed, when generating the plain text
        representation of the message.  MIME headers (MIME-Version,
        Content-Type, Content-Transfer-Encoding) will be added as needed.
        """
    if charset is None:
        self.del_param('charset')
        self._charset = None
        return
    if not isinstance(charset, Charset):
        charset = Charset(charset)
    self._charset = charset
    if 'MIME-Version' not in self:
        self.add_header('MIME-Version', '1.0')
    if 'Content-Type' not in self:
        self.add_header('Content-Type', 'text/plain', charset=charset.get_output_charset())
    else:
        self.set_param('charset', charset.get_output_charset())
    if charset != charset.get_output_charset():
        self._payload = charset.body_encode(self._payload)
    if 'Content-Transfer-Encoding' not in self:
        cte = charset.get_body_encoding()
        try:
            cte(self)
        except TypeError:
            payload = self._payload
            if payload:
                try:
                    payload = payload.encode('ascii', 'surrogateescape')
                except UnicodeError:
                    payload = payload.encode(charset.output_charset)
            self._payload = charset.body_encode(payload)
            self.add_header('Content-Transfer-Encoding', cte)