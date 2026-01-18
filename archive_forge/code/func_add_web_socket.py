import asyncio
import datetime
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import _api, backend_bases, backend_tools
from matplotlib.backends import backend_agg
from matplotlib.backend_bases import (
def add_web_socket(self, web_socket):
    assert hasattr(web_socket, 'send_binary')
    assert hasattr(web_socket, 'send_json')
    self.web_sockets.add(web_socket)
    self.resize(*self.canvas.figure.bbox.size)
    self._send_event('refresh')