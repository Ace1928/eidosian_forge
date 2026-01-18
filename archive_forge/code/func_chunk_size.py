import os
from base64 import b64encode
from collections import deque
from http.client import HTTPConnection
from json import loads
from threading import Event, Thread
from time import sleep
from urllib.parse import urlparse, urlunparse
import requests
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.weakmethod import WeakMethod
@property
def chunk_size(self):
    """Return the size of a chunk, used only in "progress" mode (when
        on_progress callback is set.)
        """
    return self._chunk_size