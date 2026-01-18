import base64
import collections
import contextlib
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import timeit
from ._interfaces import Model
import six
from tensorflow.python.framework import dtypes  # pylint: disable=g-direct-tensorflow-import
def parse_instances(request_json):
    """Parses instances from the json request sent to prediction server.

  Args:
    request_json(Text): The JSON formatted request to parse.

  Returns:
    Instances from the request json.

  Raises:
    ValueError if request_json is malformed.
  """
    if not isinstance(request_json, collections_lib.Mapping):
        raise ValueError('Invalid request sent to prediction server: {}'.format(repr(request_json)))
    if INSTANCES_KEY not in request_json:
        raise ValueError("Required field '{}' missing in prediction server request: {}".format(INSTANCES_KEY, repr(request_json)))
    return request_json.pop(INSTANCES_KEY)