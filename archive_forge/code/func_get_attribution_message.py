import base64
import io
from typing import Dict, Union
import PIL
import torch
from parlai.core.image_featurizers import ImageLoader
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.typing import TShared
def get_attribution_message(self):
    return f'This image was originally under the license *{self._image_license}({self._image_license_url})* by user *{self._username}*. The original image link is *{self._image_url}*.'