import re
import inspect
import traceback
import copy
import logging
import hmac
from base64 import b64decode
import tornado
from ..utils import template, bugreport, strtobool
@property
def capp(self):
    """return Celery application object"""
    return self.application.capp