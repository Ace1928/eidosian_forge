from urllib import parse
from zunclient.common import base
from zunclient.common import utils
def force_down(self, host, binary, force_down=None):
    """Force service state to down specified by hostname and binary."""
    body = self._update_body(host, binary, force_down=force_down)
    return self._action('/force_down', qparams=body)