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
def should_base64_decode(framework, model, signature_name):
    """Determines if base64 decoding is required.

  Returns False if framework is not TF.
  Returns True if framework is TF and is a user model.
  Returns True if framework is TF and model contains a str input.
  Returns False if framework is TF and model does not contain str input.

  Args:
    framework: ML framework of prediction app
    model: model object
    signature_name: str of name of signature

  Returns:
    bool

  """
    return framework == TENSORFLOW_FRAMEWORK_NAME and (not isinstance(model, BaseModel) or does_signature_contain_str(model.get_signature(signature_name)[1]))