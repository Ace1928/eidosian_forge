from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np
import PIL.Image
from gradio import processing_utils
def format_image(im: PIL.Image.Image | None, type: Literal['numpy', 'pil', 'filepath'], cache_dir: str, name: str='image', format: str='webp') -> np.ndarray | PIL.Image.Image | str | None:
    """Helper method to format an image based on self.type"""
    if im is None:
        return im
    if type == 'pil':
        return im
    elif type == 'numpy':
        return np.array(im)
    elif type == 'filepath':
        try:
            path = processing_utils.save_pil_to_cache(im, cache_dir=cache_dir, name=name, format=format)
        except (KeyError, ValueError):
            path = processing_utils.save_pil_to_cache(im, cache_dir=cache_dir, name=name, format='png')
        return path
    else:
        raise ValueError('Unknown type: ' + str(type) + ". Please choose from: 'numpy', 'pil', 'filepath'.")