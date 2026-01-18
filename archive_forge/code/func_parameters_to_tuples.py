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
def parameters_to_tuples(self, params, collection_formats):
    """
        Get parameters as list of tuples, formatting collections.

        :param params: Parameters as dict or list of two-tuples
        :param dict collection_formats: Parameter collection formats
        :return: Parameters as list of tuples, collections formatted
        """
    new_params = []
    if collection_formats is None:
        collection_formats = {}
    for k, v in iteritems(params) if isinstance(params, dict) else params:
        if k in collection_formats:
            collection_format = collection_formats[k]
            if collection_format == 'multi':
                new_params.extend(((k, value) for value in v))
            else:
                if collection_format == 'ssv':
                    delimiter = ' '
                elif collection_format == 'tsv':
                    delimiter = '\t'
                elif collection_format == 'pipes':
                    delimiter = '|'
                else:
                    delimiter = ','
                new_params.append((k, delimiter.join((str(value) for value in v))))
        else:
            new_params.append((k, v))
    return new_params