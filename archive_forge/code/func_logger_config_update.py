import logging
import os
import sys
from functools import partial
import pathlib
import kivy
from kivy.utils import platform
def logger_config_update(section, key, value):
    if KIVY_LOG_MODE != 'PYTHON':
        if LOG_LEVELS.get(value) is None:
            raise AttributeError("Loglevel {0!r} doesn't exists".format(value))
        Logger.setLevel(level=LOG_LEVELS.get(value))