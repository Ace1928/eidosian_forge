import abc
import base64
import functools
import hashlib
import json
import threading
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
def get_cache_id(self):
    """Fetch an identifier that uniquely identifies the auth options.

        The returned identifier need not be decomposable or otherwise provide
        any way to recreate the plugin.

        This string MUST change if any of the parameters that are used to
        uniquely identity this plugin change. It should not change upon a
        reauthentication of the plugin.

        :returns: A unique string for the set of options
        :rtype: str or None if this is unsupported or unavailable.
        """
    try:
        elements = self.get_cache_id_elements()
    except NotImplementedError:
        return None
    hasher = hashlib.sha256()
    for k, v in sorted(elements.items()):
        if v is not None:
            if isinstance(k, str):
                k = k.encode('utf-8')
            if isinstance(v, str):
                v = v.encode('utf-8')
            hasher.update(k)
            hasher.update(v)
    return base64.b64encode(hasher.digest()).decode('utf-8')