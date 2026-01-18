from __future__ import annotations
import warnings
from pathlib import Path
from typing import Any, Literal, cast
import numpy as np
import PIL.Image
from gradio_client import file
from gradio_client.documentation import document
from PIL import ImageOps
from gradio import image_utils, utils
from gradio.components.base import Component, StreamingInput
from gradio.data_classes import FileData
from gradio.events import Events
def check_streamable(self):
    if self.streaming and self.sources != ['webcam']:
        raise ValueError("Image streaming only available if sources is ['webcam']. Streaming not supported with multiple sources.")