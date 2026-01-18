import os
import numpy as np
from imageio import imread
from ..VideoClip import VideoClip
def find_image_index(t):
    return max([i for i in range(len(self.sequence)) if self.images_starts[i] <= t])