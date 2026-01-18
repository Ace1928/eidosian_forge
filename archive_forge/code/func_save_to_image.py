import os
import xarray as xr
import datashader as ds
import pandas as pd
import numpy as np
import pytest
def save_to_image(img, filename):
    """Save a shaded image as PNG file.

    Parameters
    ----------
    img: xarray
      The image to save
    filename: unicode
      The name of the file to save to, 'png' extension will be appended.

    """
    filename = os.path.join(datadir, filename + '.png')
    print('Saving: ' + filename)
    img.to_pil().save(filename)