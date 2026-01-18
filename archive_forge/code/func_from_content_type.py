from io import StringIO
from mimetypes import MimeTypes
from pkgutil import get_data
from typing import Dict, Mapping, Optional, Type, Union
from scrapy.http import Response
from scrapy.utils.misc import load_object
from scrapy.utils.python import binary_is_text, to_bytes, to_unicode
def from_content_type(self, content_type: Union[str, bytes], content_encoding: Optional[bytes]=None) -> Type[Response]:
    """Return the most appropriate Response class from an HTTP Content-Type
        header"""
    if content_encoding:
        return Response
    mimetype = to_unicode(content_type, encoding='latin-1').split(';')[0].strip().lower()
    return self.from_mimetype(mimetype)