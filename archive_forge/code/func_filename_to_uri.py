import os
import posixpath
import re
import stat
import threading
from mako import exceptions
from mako import util
from mako.template import Template
def filename_to_uri(self, filename):
    """Convert the given ``filename`` to a URI relative to
        this :class:`.TemplateCollection`."""
    try:
        return self._uri_cache[filename]
    except KeyError:
        value = self._relativeize(filename)
        self._uri_cache[filename] = value
        return value