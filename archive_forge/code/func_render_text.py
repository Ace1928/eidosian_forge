import io
import math
from typing import Dict, Optional, Union
import numpy as np
from huggingface_hub import hf_hub_download
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import convert_to_rgb, normalize, to_channel_dimension_format, to_pil_image
from ...image_utils import (
from ...utils import TensorType, is_torch_available, is_vision_available, logging
from ...utils.import_utils import requires_backends
def render_text(text: str, text_size: int=36, text_color: str='black', background_color: str='white', left_padding: int=5, right_padding: int=5, top_padding: int=5, bottom_padding: int=5, font_bytes: Optional[bytes]=None, font_path: Optional[str]=None) -> Image.Image:
    """
    Render text. This script is entirely adapted from the original script that can be found here:
    https://github.com/google-research/pix2struct/blob/main/pix2struct/preprocessing/preprocessing_utils.py

    Args:
        text (`str`, *optional*, defaults to ):
            Text to render.
        text_size (`int`, *optional*, defaults to 36):
            Size of the text.
        text_color (`str`, *optional*, defaults to `"black"`):
            Color of the text.
        background_color (`str`, *optional*, defaults to `"white"`):
            Color of the background.
        left_padding (`int`, *optional*, defaults to 5):
            Padding on the left.
        right_padding (`int`, *optional*, defaults to 5):
            Padding on the right.
        top_padding (`int`, *optional*, defaults to 5):
            Padding on the top.
        bottom_padding (`int`, *optional*, defaults to 5):
            Padding on the bottom.
        font_bytes (`bytes`, *optional*):
            Bytes of the font to use. If `None`, the default font will be used.
        font_path (`str`, *optional*):
            Path to the font to use. If `None`, the default font will be used.
    """
    requires_backends(render_text, 'vision')
    wrapper = textwrap.TextWrapper(width=80)
    lines = wrapper.wrap(text=text)
    wrapped_text = '\n'.join(lines)
    if font_bytes is not None and font_path is None:
        font = io.BytesIO(font_bytes)
    elif font_path is not None:
        font = font_path
    else:
        font = hf_hub_download(DEFAULT_FONT_PATH, 'Arial.TTF')
    font = ImageFont.truetype(font, encoding='UTF-8', size=text_size)
    temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1), background_color))
    _, _, text_width, text_height = temp_draw.textbbox((0, 0), wrapped_text, font)
    image_width = text_width + left_padding + right_padding
    image_height = text_height + top_padding + bottom_padding
    image = Image.new('RGB', (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)
    draw.text(xy=(left_padding, top_padding), text=wrapped_text, fill=text_color, font=font)
    return image