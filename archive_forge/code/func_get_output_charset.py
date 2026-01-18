from functools import partial
import email.base64mime
import email.quoprimime
from email import errors
from email.encoders import encode_7or8bit
def get_output_charset(self):
    """Return the output character set.

        This is self.output_charset if that is not None, otherwise it is
        self.input_charset.
        """
    return self.output_charset or self.input_charset