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
def refresh_all(self):
    if self.web_sockets:
        diff = self.canvas.get_diff_image()
        if diff is not None:
            for s in self.web_sockets:
                s.send_binary(diff)