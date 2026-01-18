import ctypes
import json
import logging
import os
import sys
import cffi  # type: ignore
import six
from google.auth import exceptions
def attach_to_ssl_context(self, ctx):
    if not self._offload_lib.ConfigureSslContext(self._sign_callback, ctypes.c_char_p(self._cert), _cast_ssl_ctx_to_void_p(ctx._ctx._context)):
        raise exceptions.MutualTLSChannelError('failed to configure SSL context')