import skvideo.io
import skvideo.utils
import numpy as np
import os
import sys
def pattern_noise(backend):
    np.random.seed(1)
    randomNoiseData = np.random.random((100, 100)) * 255
    randomNoiseData[0, 0] = 0
    randomNoiseData[0, 1] = 1
    randomNoiseData[0, 2] = 255
    skvideo.io.vwrite('randomNoisePattern.yuv', randomNoiseData, backend=backend)
    videoData1 = skvideo.io.vread('randomNoisePattern.yuv', width=100, height=100, backend=backend)
    skvideo.io.vwrite('randomNoisePattern_resaved.yuv', videoData1, backend=backend)
    videoData2 = skvideo.io.vread('randomNoisePattern_resaved.yuv', width=100, height=100, backend=backend)
    randomDataOriginal = np.array(randomNoiseData)
    randomDataVideo1 = skvideo.utils.rgb2gray(videoData1[0])[0, :, :, 0]
    randomDataVideo2 = skvideo.utils.rgb2gray(videoData2[0])[0, :, :, 0]
    floattopixel_mse = np.mean((randomDataOriginal - randomDataVideo1) ** 2)
    assert floattopixel_mse < 1, 'Possible conversion error between floating point and raw video. MSE=%f' % (floattopixel_mse,)
    pixeltopixel_mse = np.mean((randomDataVideo1 - randomDataVideo2) ** 2)
    assert pixeltopixel_mse == 0, 'Creeping error inside vread/vwrite.'
    os.remove('randomNoisePattern.yuv')
    os.remove('randomNoisePattern_resaved.yuv')