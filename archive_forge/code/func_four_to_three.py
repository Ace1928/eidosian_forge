import numpy as np
from .loadsave import load
from .orientations import OrientationError, io_orientation
def four_to_three(img):
    """Create 3D images from 4D image by slicing over last axis

    Parameters
    ----------
    img :  image
       4D image instance of some class with methods ``get_data``,
       ``header`` and ``affine``, and a class constructor
       allowing klass(data, affine, header)

    Returns
    -------
    imgs : list
       list of 3D images
    """
    arr = np.asanyarray(img.dataobj)
    header = img.header
    affine = img.affine
    image_maker = img.__class__
    if arr.ndim != 4:
        raise ValueError('Expecting four dimensions')
    imgs = []
    for i in range(arr.shape[3]):
        arr3d = arr[..., i]
        img3d = image_maker(arr3d, affine, header)
        imgs.append(img3d)
    return imgs