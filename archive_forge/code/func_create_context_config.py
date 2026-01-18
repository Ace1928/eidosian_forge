from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
import json
import os
from boto import config
import gslib
from gslib import exception
from gslib.utils import boto_util
from gslib.utils import execution_util
def create_context_config(logger):
    """Should be run once at gsutil startup. Creates global singleton.

  Args:
    logger (logging.logger): For logging during config functions.

  Returns:
    New ContextConfig singleton.

  Raises:
    Exception if singleton already exists.
  """
    global _singleton_config
    if not _singleton_config:
        _singleton_config = _ContextConfig(logger)
        return _singleton_config
    raise ContextConfigSingletonAlreadyExistsError