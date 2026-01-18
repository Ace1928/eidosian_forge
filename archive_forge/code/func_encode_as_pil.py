import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
@staticmethod
def encode_as_pil(obj):
    """Attempt to convert PIL.Image.Image to base64 data uri"""
    image = get_module('PIL.Image')
    if image is not None and isinstance(obj, image.Image):
        return ImageUriValidator.pil_image_to_uri(obj)
    else:
        raise NotEncodable