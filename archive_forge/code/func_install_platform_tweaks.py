import logging
import os
import platform as _platform
import sys
from datetime import datetime
from functools import partial
from billiard.common import REMAP_SIGTERM
from billiard.process import current_process
from kombu.utils.encoding import safe_str
from celery import VERSION_BANNER, platforms, signals
from celery.app import trace
from celery.loaders.app import AppLoader
from celery.platforms import EX_FAILURE, EX_OK, check_privileges
from celery.utils import static, term
from celery.utils.debug import cry
from celery.utils.imports import qualname
from celery.utils.log import get_logger, in_sighandler, set_in_sighandler
from celery.utils.text import pluralize
from celery.worker import WorkController
def install_platform_tweaks(self, worker):
    """Install platform specific tweaks and workarounds."""
    if self.app.IS_macOS:
        self.macOS_proxy_detection_workaround()
    if not self._isatty:
        if self.app.IS_macOS:
            install_HUP_not_supported_handler(worker)
        else:
            install_worker_restart_handler(worker)
    install_worker_term_handler(worker)
    install_worker_term_hard_handler(worker)
    install_worker_int_handler(worker)
    install_cry_handler()
    install_rdb_handler()