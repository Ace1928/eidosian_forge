import numpy as np
import scipy.fftpack
import scipy.io
import scipy.misc
import scipy.ndimage
import scipy.stats
from ..utils import *
def brisque_features(videoData):
    """Computes Blind/Referenceless Image Spatial QUality Evaluator (BRISQUE) features. [#f1]_ [#f2]_

    Since this is a referenceless image quality algorithm, only 1 video is needed. This function
    provides the raw features extracted per frame.

    Parameters
    ----------
    videoData : ndarray
        Reference video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels.

    Returns
    -------
    features : ndarray
        A matrix of shape (T, 36) of the computed BRISQUE features.

    References
    ----------

    .. [#f1] A. Mittal, A. K. Moorthy and A. C. Bovik, "No-Reference Image Quality Assessment in the Spatial Domain" IEEE Transactions on Image Processing, 2012. 
    .. [#f2] A. Mittal, A. K. Moorthy and A. C. Bovik, "Referenceless Image Spatial Quality Evaluation Engine," 45th Asilomar Conference on Signals, Systems and Computers , November 2011.

    """
    videoData = vshape(videoData)
    T, M, N, C = videoData.shape
    assert C == 1, 'brisque_features called with video having %d channels. Please supply only the luminance channel.' % (C,)
    feats = np.zeros((T, 36), dtype=np.float32)
    for i in range(T):
        full_scale = videoData[i, :, :, 0].astype(np.float32)
        half_scale = scipy.misc.imresize(full_scale, 0.5, interp='bicubic', mode='F')
        full_scale, _, _ = compute_image_mscn_transform(full_scale)
        half_scale, _, _ = compute_image_mscn_transform(half_scale)
        feats[i, 18 * 0:18 * 1] = _extract_subband_feats(full_scale)
        feats[i, 18 * 1:18 * 2] = _extract_subband_feats(half_scale)
    return feats