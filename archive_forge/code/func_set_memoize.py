from copy import copy
import numpy as np
import proglog
from tqdm import tqdm
from moviepy.decorators import (apply_to_audio, apply_to_mask,
@outplace
def set_memoize(self, memoize):
    """ Sets wheter the clip should keep the last frame read in memory """
    self.memoize = memoize