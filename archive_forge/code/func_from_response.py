import sys
import email.parser
from .encoder import encode_with
from requests.structures import CaseInsensitiveDict
@classmethod
def from_response(cls, response, encoding='utf-8'):
    content = response.content
    content_type = response.headers.get('content-type', None)
    return cls(content, content_type, encoding)