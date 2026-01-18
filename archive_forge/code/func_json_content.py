import codecs
import functools
import json
import os
import traceback
from testtools.compat import _b
from testtools.content_type import ContentType, JSON, UTF8_TEXT
def json_content(json_data):
    """Create a JSON Content object from JSON-encodeable data."""
    data = json.dumps(json_data)
    data = data.encode('utf8')
    return Content(JSON, lambda: [data])