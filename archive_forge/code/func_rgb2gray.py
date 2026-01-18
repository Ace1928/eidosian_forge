import logging
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
@DeveloperAPI
def rgb2gray(img: np.ndarray) -> np.ndarray:
    if cv2:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return color.rgb2gray(img)