import numpy as np
def flim(im):
    return hill(np.sqrt(((im - color) ** 2).sum(axis=2)))