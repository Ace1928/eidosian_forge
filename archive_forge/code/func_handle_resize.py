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
def handle_resize(self, event):
    x = int(event.get('width', 800)) * self.device_pixel_ratio
    y = int(event.get('height', 800)) * self.device_pixel_ratio
    fig = self.figure
    fig.set_size_inches(x / fig.dpi, y / fig.dpi, forward=False)
    self._png_is_old = True
    self.manager.resize(*fig.bbox.size, forward=False)
    ResizeEvent('resize_event', self)._process()
    self.draw_idle()