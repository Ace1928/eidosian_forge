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
def prepare_post_parameters(self, post_params=None, files=None):
    """
        Builds form parameters.

        :param post_params: Normal form parameters.
        :param files: File parameters.
        :return: Form parameters with files.
        """
    params = []
    if post_params:
        params = post_params
    if files:
        for k, v in iteritems(files):
            if not v:
                continue
            file_names = v if type(v) is list else [v]
            for n in file_names:
                with open(n, 'rb') as f:
                    filename = os.path.basename(f.name)
                    filedata = f.read()
                    mimetype = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
                    params.append(tuple([k, tuple([filename, filedata, mimetype])]))
    return params