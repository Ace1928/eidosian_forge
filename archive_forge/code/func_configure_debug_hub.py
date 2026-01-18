import sys
import logging
from sentry_sdk import utils
from sentry_sdk.hub import Hub
from sentry_sdk.utils import logger
from sentry_sdk.client import _client_init_debug
from logging import LogRecord
def configure_debug_hub():

    def _get_debug_hub():
        return Hub.current
    utils._get_debug_hub = _get_debug_hub