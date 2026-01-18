import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def skin():
    """Microscopy image of dermis and epidermis (skin layers).

    Hematoxylin and eosin stained slide at 10x of normal epidermis and dermis
    with a benign intradermal nevus.

    Notes
    -----
    This image requires an Internet connection the first time it is called,
    and to have the ``pooch`` package installed, in order to fetch the image
    file from the scikit-image datasets repository.

    The source of this image is
    https://en.wikipedia.org/wiki/File:Normal_Epidermis_and_Dermis_with_Intradermal_Nevus_10x.JPG

    The image was released in the public domain by its author Kilbad.

    Returns
    -------
    skin : (960, 1280, 3) RGB image of uint8
    """
    return _load('data/skin.jpg')