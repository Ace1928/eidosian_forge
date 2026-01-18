import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def pack_content(self, filename, content):
    enc_filename = self.base64encode(filename) or '-'
    enc_content = (content or '').encode('base64')
    result = '%s %s' % (enc_filename, enc_content)
    return result