import os
from typing import List, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
def large_separated_enc_dec_w_lyrics(layer):
    return _LARGE_ATTENTION[layer % 79]