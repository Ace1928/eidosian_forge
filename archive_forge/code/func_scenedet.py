import numpy as np
import scipy.ndimage
import scipy.spatial
from ..motion.gme import globalEdgeMotion
from ..utils import *
def scenedet(videodata, method='histogram', parameter1=None, min_scene_length=2):
    """Scene detection algorithms
    
    Given a sequence of frames, this function
    is able to run find the first index of new
    scenes.

    Parameters
    ----------
    videodata : ndarray
        an input frame sequence, shape (T, M, N, C), (T, M, N), (M, N, C) or (M, N)

    method : string
        "histogram" --> threshold-based (parameter1 defaults to 1.0) approach using intensity histogram differences. [#f1]_

        "edges" --> threshold-based (parameter1 defaults to 0.5) approach measuring the edge-change fraction after global motion compensation [#f2]_

        "intensity" --> Detects fast cuts using changes in colour and intensity between frames. Parameter1 is the threshold used for detection, which defaults to 30.0.

    parameter1 : float
        Number used as a tuning parameter. See method argument for details.

    min_scene_length : int
        Number used for determining minimum scene length.

    Returns
    ----------
    sceneCuts : ndarray, shape (numScenes,)

        The indices corresponding to the first frame in the detected scenes.

    References
    ----------
    .. [#f1] Kiyotaka Otsuji and Yoshinobu Tonomura. Projection-detecting filter for video cut detection. Multimedia Systems 1.5, 205-210, 1994.

    .. [#f2] Kevin Mai, Ramin Zabih, and Justin Miller. Feature-based algorithms for detecting and classifying scene breaks. Cornell University, 1995.

    """
    videodata = vshape(videodata)
    detected_scenes = []
    if method == 'histogram':
        if parameter1 is None:
            parameter1 = 1.0
        detected_scenes = _scenedet_histogram(videodata, parameter1, min_scene_length)
    elif method == 'edges':
        if parameter1 is None:
            parameter1 = 0.6
        detected_scenes = _scenedet_edges(videodata, parameter1, min_scene_length)
    elif method == 'intensity':
        if parameter1 is None:
            parameter1 = 1.0
        detected_scenes = _scenedet_intensity(videodata, parameter1, min_scene_length)
    else:
        raise NotImplementedError
    return detected_scenes