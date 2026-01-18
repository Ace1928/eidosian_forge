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
def does_signature_contain_str(signature=None):
    """Return true if input signature contains a string dtype.

  This is used to determine if we should proceed with base64 decoding.

  Args:
    signature: SignatureDef protocol buffer

  Returns:
    bool
  """
    if signature is None:
        return True
    return any((v.dtype == dtypes.string.as_datatype_enum for v in signature.inputs.values()))