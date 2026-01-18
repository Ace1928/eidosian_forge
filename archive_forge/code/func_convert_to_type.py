from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Union
from urllib.parse import urlparse
import numpy as np
import PIL.Image
from gradio_client import file
from gradio_client.documentation import document
from gradio_client.utils import is_http_url_like
from gradio import processing_utils, utils, wasm_utils
from gradio.components.base import Component
from gradio.data_classes import FileData, GradioModel, GradioRootModel
from gradio.events import Events
@staticmethod
def convert_to_type(img: str, type: Literal['filepath', 'numpy', 'pil']):
    if type == 'filepath':
        return img
    else:
        converted_image = PIL.Image.open(img)
        if type == 'numpy':
            converted_image = np.array(converted_image)
        return converted_image