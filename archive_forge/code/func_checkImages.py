import os
import zlib
import time  # noqa
import logging
import numpy as np
def checkImages(images):
    """checkImages(images)
    Check numpy images and correct intensity range etc.
    The same for all movie formats.
    """
    images2 = []
    for im in images:
        if isinstance(im, np.ndarray):
            if im.dtype == np.uint8:
                images2.append(im)
            elif im.dtype in [np.float32, np.float64]:
                theMax = im.max()
                if 128 < theMax < 300:
                    pass
                else:
                    im = im.copy()
                    im[im < 0] = 0
                    im[im > 1] = 1
                    im *= 255
                images2.append(im.astype(np.uint8))
            else:
                im = im.astype(np.uint8)
                images2.append(im)
            if im.ndim == 2:
                pass
            elif im.ndim == 3:
                if im.shape[2] not in [3, 4]:
                    raise ValueError('This array can not represent an image.')
            else:
                raise ValueError('This array can not represent an image.')
        else:
            raise ValueError('Invalid image type: ' + str(type(im)))
    return images2