from ..utils import *
import numpy as np
import scipy.ndimage
def ssim_full(referenceVideoData, distortedVideoData, K_1=0.01, K_2=0.03, bitdepth=8, scaleFix=True, avg_window=None):
    """Returns all parameters from the Structural Similarity (SSIM) Index. [#f1]_

    Both video inputs are compared frame-by-frame to obtain T
    SSIM measurements on the luminance channel.

    Parameters
    ----------
    referenceVideoData : ndarray
        Reference video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels. Here C is only allowed to be 1.

    distortedVideoData : ndarray
        Distorted video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels. Here C is only allowed to be 1.

    K_1 : float
        Luminance saturation weight

    K_2 : float
        Contrast saturation weight

    bitdepth : int
        The number of bits each pixel effectively has

    scaleFix : bool
        Whether to scale the input frame size based on assumed distance, to improve subjective correlation.

    avg_window : ndarray
        2-d averaging window, normalized to unit volume.

    Returns
    -------
    ssim_array : ndarray
        The ssim results, ndarray of dimension (T,), where T
        is the number of frames

    ssim_map_array : ndarray
        The ssim maps, ndarray of dimension (T,M-10, N-10), where T
        is the number of frames, and MxN are the widthxheight

    contrast_array : ndarray
        The ssim result based on only on contrast (no luminance masking),
        ndarray of dimension (T,), where T is the number of frames

    contrast_map_array : ndarray
        The ssim contrast-only maps, ndarray of dimension (T,M-10, N-10), where T
        is the number of frames, and MxN are the widthxheight


    References
    ----------

    .. [#f1] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error measurement to structural similarity" IEEE Transactions on Image Processing, vol. 13, no. 1, Jan. 2004.

    """
    referenceVideoData = vshape(referenceVideoData)
    distortedVideoData = vshape(distortedVideoData)
    assert referenceVideoData.shape == distortedVideoData.shape
    T, M, N, C = referenceVideoData.shape
    assert C == 1, 'ssim called with videos containing %d channels. Please supply only the luminance channel' % (C,)
    ssim_maps = np.zeros((T, M - 10, N - 10), dtype=np.float32)
    contrast_maps = np.zeros((T, M - 10, N - 10), dtype=np.float32)
    ssim_scores = np.zeros(T, dtype=np.float32)
    contrast_scores = np.zeros(T, dtype=np.float32)
    for t in range(T):
        mssim, ssim_map, mcs, cs_map = _ssim_core(referenceVideoData[t, :, :, 0], distortedVideoData[t, :, :, 0], K_1=K_1, K_2=K_2, bitdepth=bitdepth, scaleFix=scaleFix, avg_window=avg_window)
        ssim_scores[t] = mssim
        contrast_scores[t] = mcs
        ssim_maps[t] = ssim_map
        contrast_maps[t] = cs_map
    return (ssim_scores, ssim_maps, contrast_scores, contrast_maps)