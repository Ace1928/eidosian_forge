import sys
import email.parser
from .encoder import encode_with
from requests.structures import CaseInsensitiveDict
def body_part(part):
    fixed = MultipartDecoder._fix_first_part(part, boundary)
    return BodyPart(fixed, self.encoding)