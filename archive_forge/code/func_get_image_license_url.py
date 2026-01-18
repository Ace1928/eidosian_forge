import base64
import io
from typing import Dict, Union
import PIL
import torch
from parlai.core.image_featurizers import ImageLoader
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.typing import TShared
def get_image_license_url(self):
    return self._image_license_url