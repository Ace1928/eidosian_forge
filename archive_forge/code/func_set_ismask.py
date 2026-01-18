from copy import copy
import numpy as np
import proglog
from tqdm import tqdm
from moviepy.decorators import (apply_to_audio, apply_to_mask,
@outplace
def set_ismask(self, ismask):
    """ Says wheter the clip is a mask or not (ismask is a boolean)"""
    self.ismask = ismask