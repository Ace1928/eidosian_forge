from __future__ import annotations
import errno
import itertools
import mimetypes
import os
import time
import warnings
from html import escape
from typing import Any, Callable, Dict, Sequence
from urllib.parse import quote, unquote
from zope.interface import implementer
from incremental import Version
from typing_extensions import Literal
from twisted.internet import abstract, interfaces
from twisted.python import components, filepath, log
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecated
from twisted.python.runtime import platformType
from twisted.python.url import URL
from twisted.python.util import InsensitiveDict
from twisted.web import http, resource, server
from twisted.web.util import redirectTo
def loadMimeTypes(mimetype_locations=None, init=mimetypes.init):
    """
    Produces a mapping of extensions (with leading dot) to MIME types.

    It does this by calling the C{init} function of the L{mimetypes} module.
    This will have the side effect of modifying the global MIME types cache
    in that module.

    Multiple file locations containing mime-types can be passed as a list.
    The files will be sourced in that order, overriding mime-types from the
    files sourced beforehand, but only if a new entry explicitly overrides
    the current entry.

    @param mimetype_locations: Optional. List of paths to C{mime.types} style
        files that should be used.
    @type mimetype_locations: iterable of paths or L{None}
    @param init: The init function to call. Defaults to the global C{init}
        function of the C{mimetypes} module. For internal use (testing) only.
    @type init: callable
    """
    init(mimetype_locations)
    mimetypes.types_map.update({'.conf': 'text/plain', '.diff': 'text/plain', '.flac': 'audio/x-flac', '.java': 'text/plain', '.oz': 'text/x-oz', '.swf': 'application/x-shockwave-flash', '.wml': 'text/vnd.wap.wml', '.xul': 'application/vnd.mozilla.xul+xml', '.patch': 'text/plain'})
    return mimetypes.types_map