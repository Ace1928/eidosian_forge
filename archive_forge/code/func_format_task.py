import re
import inspect
import traceback
import copy
import logging
import hmac
from base64 import b64decode
import tornado
from ..utils import template, bugreport, strtobool
def format_task(self, task):
    custom_format_task = self.application.options.format_task
    if custom_format_task:
        try:
            task = custom_format_task(copy.copy(task))
        except Exception:
            logger.exception("Failed to format '%s' task", task.uuid)
    return task