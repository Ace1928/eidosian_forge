import logging
import os
import sys
import time
from typing import Any, Dict
from billiard.einfo import ExceptionInfo
from billiard.exceptions import WorkerLostError
from kombu.utils.encoding import safe_repr
from celery.exceptions import WorkerShutdown, WorkerTerminate, reraise
from celery.utils import timer2
from celery.utils.log import get_logger
from celery.utils.text import truncate
def on_soft_timeout(self, job):
    pass