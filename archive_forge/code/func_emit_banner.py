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
def emit_banner(self):
    use_image = term.supports_images()
    if use_image:
        print(term.imgcat(static.logo()))
    print(safe_str(''.join([str(self.colored.cyan(' \n', self.startup_info(artlines=not use_image))), str(self.colored.reset(self.extra_info() or ''))])), file=sys.__stdout__, flush=True)