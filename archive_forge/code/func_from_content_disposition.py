from io import StringIO
from mimetypes import MimeTypes
from pkgutil import get_data
from typing import Dict, Mapping, Optional, Type, Union
from scrapy.http import Response
from scrapy.utils.misc import load_object
from scrapy.utils.python import binary_is_text, to_bytes, to_unicode
def from_content_disposition(self, content_disposition: Union[str, bytes]) -> Type[Response]:
    try:
        filename = to_unicode(content_disposition, encoding='latin-1', errors='replace').split(';')[1].split('=')[1].strip('"\'')
        return self.from_filename(filename)
    except IndexError:
        return Response