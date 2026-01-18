import re
import binascii
import email.quoprimime
import email.base64mime
from email.errors import HeaderParseError
from email import charset as _charset
def part_count(self):
    return super().__len__()