import functools
import json
import multiprocessing
import os
import threading
from contextlib import contextmanager
from threading import Thread
from ._colorizer import Colorizer
from ._locks_machinery import create_handler_lock
def prepare_colored_format(format_, ansi_level):
    colored = Colorizer.prepare_format(format_)
    return (colored, colored.colorize(ansi_level))