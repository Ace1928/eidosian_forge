from kombu.utils.limits import TokenBucket
from celery import platforms
from celery.app import app_or_default
from celery.utils.dispatch import Signal
from celery.utils.imports import instantiate
from celery.utils.log import get_logger
from celery.utils.time import rate
from celery.utils.timer2 import Timer
def shutter(self):
    if self.maxrate is None or self.maxrate.can_consume():
        logger.debug('Shutter: %s', self.state)
        self.shutter_signal.send(sender=self.state)
        self.on_shutter(self.state)