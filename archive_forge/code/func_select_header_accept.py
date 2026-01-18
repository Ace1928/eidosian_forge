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
def select_header_accept(self, accepts):
    """
        Returns `Accept` based on an array of accepts provided.

        :param accepts: List of headers.
        :return: Accept (e.g. application/json).
        """
    if not accepts:
        return
    accepts = [x.lower() for x in accepts]
    if 'application/json' in accepts:
        return 'application/json'
    else:
        return ', '.join(accepts)