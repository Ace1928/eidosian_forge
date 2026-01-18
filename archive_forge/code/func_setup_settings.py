import os
import warnings
from celery.exceptions import NotConfigured
from celery.utils.collections import DictAttribute
from celery.utils.serialization import strtobool
from .base import BaseLoader
def setup_settings(self, settingsdict):
    return DictAttribute(settingsdict)