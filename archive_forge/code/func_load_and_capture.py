import importlib
import logging
import os
import sys
import types
from io import StringIO
from typing import Any, Dict, List
import breezy
from .. import osutils, plugin, tests
from . import test_bar
def load_and_capture(self, name, warn_load_problems=True):
    """Load plugins from '.' capturing the output.

        :param name: The name of the plugin.
        :return: A string with the log from the plugin loading call.
        """
    stream = StringIO()
    try:
        handler = logging.StreamHandler(stream)
        log = logging.getLogger('brz')
        log.addHandler(handler)
        try:
            self.load_with_paths(['.'], warn_load_problems=warn_load_problems)
        finally:
            handler.flush()
            handler.close()
            log.removeHandler(handler)
        return stream.getvalue()
    finally:
        stream.close()