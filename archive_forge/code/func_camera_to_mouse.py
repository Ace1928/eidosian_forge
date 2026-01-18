import json
import os
import numpy as np
def camera_to_mouse(camera_ac):
    """
    Convert camera angles (pitch, yaw) (minerl format) to mouse movement (dx, dy) (minerec format)
    """
    return {'dx': camera_ac[1] / CAMERA_SCALER, 'dy': camera_ac[0] / CAMERA_SCALER}