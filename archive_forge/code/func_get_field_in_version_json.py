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
def get_field_in_version_json(field_name):
    """Gets the value of field_name in the version being created, if it exists.

  Args:
    field_name: Name of the key used for retrieving the corresponding value from
      version json object.

  Returns:
  The value of the given field in the version object or the user provided create
  version request if it exists. Otherwise None is returned.
  """
    if not os.environ.get('create_version_request'):
        return None
    request = json.loads(os.environ.get('create_version_request'))
    if not request or not isinstance(request, dict):
        return None
    version = request.get('version')
    if not version or not isinstance(version, dict):
        return None
    logging.info('Found value: %s, for field: %s from create_version_request', version.get(field_name), field_name)
    return version.get(field_name)