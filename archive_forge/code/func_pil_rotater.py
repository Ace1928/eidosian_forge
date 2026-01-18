import numpy as np
from moviepy.decorators import apply_to_mask
def pil_rotater(pic, angle, resample, expand):
    return np.array(Image.fromarray(pic).rotate(angle, expand=expand, resample=resample))