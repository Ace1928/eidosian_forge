from __future__ import absolute_import
import os
import re
import json
import mimetypes
import tempfile
from multiprocessing.pool import ThreadPool
from datetime import date, datetime
from six import PY3, integer_types, iteritems, text_type
from six.moves.urllib.parse import quote
from . import models
from .configuration import Configuration
from .rest import ApiException, RESTClientObject
def select_header_content_type(self, content_types):
    """
        Returns `Content-Type` based on an array of content_types provided.

        :param content_types: List of content-types.
        :return: Content-Type (e.g. application/json).
        """
    if not content_types:
        return 'application/json'
    content_types = [x.lower() for x in content_types]
    if 'application/json' in content_types or '*/*' in content_types:
        return 'application/json'
    else:
        return content_types[0]