from _pydevd_bundle.pydevd_constants import DebugInfoHolder, \
from _pydevd_bundle.pydevd_utils import quote_smart as quote, to_string
from _pydevd_bundle.pydevd_comm_constants import ID_TO_MEANING, CMD_EXIT
from _pydevd_bundle.pydevd_constants import HTTP_PROTOCOL, HTTP_JSON_PROTOCOL, \
import json
from _pydev_bundle import pydev_log
def call_after_send(self, callback):
    if not self._after_send:
        self._after_send = [callback]
    else:
        self._after_send.append(callback)