import base64
import collections
import hashlib
import io
import json
import re
import textwrap
import time
from urllib import parse as urlparse
import zipfile
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import auth_context_middleware
from tensorboard.backend import client_feature_flags
from tensorboard.backend import empty_path_redirect
from tensorboard.backend import experiment_id
from tensorboard.backend import experimental_plugin
from tensorboard.backend import http_util
from tensorboard.backend import path_prefix
from tensorboard.backend import security_validator
from tensorboard.plugins import base_plugin
from tensorboard.plugins.core import core_plugin
from tensorboard.util import tb_logging
def make_plugin_loader(plugin_spec):
    """Returns a plugin loader for the given plugin.

    Args:
      plugin_spec: A TBPlugin subclass, or a TBLoader instance or subclass.

    Returns:
      A TBLoader for the given plugin.

    :type plugin_spec:
      Type[base_plugin.TBPlugin] | Type[base_plugin.TBLoader] |
      base_plugin.TBLoader
    :rtype: base_plugin.TBLoader
    """
    if isinstance(plugin_spec, base_plugin.TBLoader):
        return plugin_spec
    if isinstance(plugin_spec, type):
        if issubclass(plugin_spec, base_plugin.TBLoader):
            return plugin_spec()
        if issubclass(plugin_spec, base_plugin.TBPlugin):
            return base_plugin.BasicLoader(plugin_spec)
    raise TypeError('Not a TBLoader or TBPlugin subclass: %r' % (plugin_spec,))