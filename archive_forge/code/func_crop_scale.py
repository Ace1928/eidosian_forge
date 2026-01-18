from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np
import PIL.Image
from gradio import processing_utils
def crop_scale(img: PIL.Image.Image, final_width: int, final_height: int):
    original_width, original_height = img.size
    target_aspect_ratio = final_width / final_height
    if original_width / original_height > target_aspect_ratio:
        crop_height = original_height
        crop_width = crop_height * target_aspect_ratio
    else:
        crop_width = original_width
        crop_height = crop_width / target_aspect_ratio
    left = (original_width - crop_width) / 2
    top = (original_height - crop_height) / 2
    img_cropped = img.crop((int(left), int(top), int(left + crop_width), int(top + crop_height)))
    img_resized = img_cropped.resize((final_width, final_height))
    return img_resized