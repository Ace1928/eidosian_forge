import numpy as np
def fl(gf, t):
    tt = np.linspace(t - d, t + d, nframes)
    avg = np.mean(1.0 * np.array([gf(t_) for t_ in tt], dtype='uint16'), axis=0)
    return avg.astype('uint8')