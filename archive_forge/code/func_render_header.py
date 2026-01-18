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
def render_header(image: np.ndarray, header: str, input_data_format: Optional[Union[str, ChildProcessError]]=None, **kwargs):
    """
    Renders the input text as a header on the input image.

    Args:
        image (`np.ndarray`):
            The image to render the header on.
        header (`str`):
            The header text.
        data_format (`Union[ChannelDimension, str]`, *optional*):
            The data format of the image. Can be either "ChannelDimension.channels_first" or
            "ChannelDimension.channels_last".

    Returns:
        `np.ndarray`: The image with the header rendered.
    """
    requires_backends(render_header, 'vision')
    image = to_pil_image(image, input_data_format=input_data_format)
    header_image = render_text(header, **kwargs)
    new_width = max(header_image.width, image.width)
    new_height = int(image.height * (new_width / image.width))
    new_header_height = int(header_image.height * (new_width / header_image.width))
    new_image = Image.new('RGB', (new_width, new_height + new_header_height), 'white')
    new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
    new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))
    new_image = to_numpy_array(new_image)
    if infer_channel_dimension_format(new_image) == ChannelDimension.LAST:
        new_image = to_channel_dimension_format(new_image, ChannelDimension.LAST)
    return new_image