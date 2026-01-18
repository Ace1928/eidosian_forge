from collections import namedtuple
import numpy as np
from ...util import dtype as dtypes
from ...exposure import is_low_contrast
from ..._shared.utils import warn
from math import floor, ceil
def imshow_collection(ic, *args, **kwargs):
    """Display all images in the collection.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The `Figure` object returned by `plt.subplots`.
    """
    import matplotlib.pyplot as plt
    if len(ic) < 1:
        raise ValueError('Number of images to plot must be greater than 0')
    num_images = len(ic)
    k = (num_images * 12) ** 0.5
    r1 = max(1, floor(k / 4))
    r2 = ceil(k / 4)
    c1 = ceil(num_images / r1)
    c2 = ceil(num_images / r2)
    if abs(r1 / c1 - 0.75) < abs(r2 / c2 - 0.75):
        nrows, ncols = (r1, c1)
    else:
        nrows, ncols = (r2, c2)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    ax = np.asarray(axes).ravel()
    for n, image in enumerate(ic):
        ax[n].imshow(image, *args, **kwargs)
    kwargs['ax'] = axes
    return fig