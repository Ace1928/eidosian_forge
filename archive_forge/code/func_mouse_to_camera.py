import json
import os
import numpy as np
def mouse_to_camera(mouse_ac):
    """
    Convert mouse movement (dx, dy) (in minerec format) into camera angles (pitch, yaw) (minerl format)
    """
    return CAMERA_SCALER * np.array([mouse_ac['dy'], mouse_ac['dx']])