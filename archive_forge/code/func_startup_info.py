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
def startup_info(self, artlines=True):
    app = self.app
    concurrency = str(self.concurrency)
    appr = '{}:{:#x}'.format(app.main or '__main__', id(app))
    if not isinstance(app.loader, AppLoader):
        loader = qualname(app.loader)
        if loader.startswith('celery.loaders'):
            loader = loader[14:]
        appr += f' ({loader})'
    if self.autoscale:
        max, min = self.autoscale
        concurrency = f'{{min={min}, max={max}}}'
    pool = self.pool_cls
    if not isinstance(pool, str):
        pool = pool.__module__
    concurrency += f' ({pool.split('.')[-1]})'
    events = 'ON'
    if not self.task_events:
        events = 'OFF (enable -E to monitor tasks in this worker)'
    banner = BANNER.format(app=appr, hostname=safe_str(self.hostname), timestamp=datetime.now().replace(microsecond=0), version=VERSION_BANNER, conninfo=self.app.connection().as_uri(), results=self.app.backend.as_uri(), concurrency=concurrency, platform=safe_str(_platform.platform()), events=events, queues=app.amqp.queues.format(indent=0, indent_first=False)).splitlines()
    if artlines:
        for i, _ in enumerate(banner):
            try:
                banner[i] = ' '.join([ARTLINES[i], banner[i]])
            except IndexError:
                banner[i] = ' ' * 16 + banner[i]
    return '\n'.join(banner) + '\n'