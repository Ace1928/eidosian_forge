import logging
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
@DeveloperAPI
def imread(img_file: str) -> np.ndarray:
    if cv2:
        return cv2.imread(img_file).astype(np.float32)
    return io.imread(img_file).astype(np.float32)