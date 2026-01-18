import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def grass():
    """Grass.

    Returns
    -------
    grass : (512, 512) uint8 image
        Some grass.

    Notes
    -----
    The original image was downloaded from
    `DeviantArt <https://www.deviantart.com/linolafett/art/Grass-01-434853879>`__
    and licensed under the Creative Commons CC0 License.

    The downloaded image was cropped to include a region of ``(512, 512)``
    pixels around the top left corner, converted to grayscale, then to uint8
    prior to saving the result in PNG format.

    """
    "\n    The following code was used to obtain the final image.\n\n    >>> import sys; print(sys.version)\n    >>> import platform; print(platform.platform())\n    >>> import skimage; print(f'scikit-image version: {skimage.__version__}')\n    >>> import numpy; print(f'numpy version: {numpy.__version__}')\n    >>> import imageio; print(f'imageio version {imageio.__version__}')\n    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)\n    [GCC 7.3.0]\n    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid\n    scikit-image version: 0.16.dev0\n    numpy version: 1.16.4\n    imageio version 2.4.1\n\n    >>> import requests\n    >>> import zipfile\n    >>> url = 'https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/a407467e-4ff0-49f1-923f-c9e388e84612/d76wfef-2878b78d-5dce-43f9-be36-26ec9bc0df3b.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2E0MDc0NjdlLTRmZjAtNDlmMS05MjNmLWM5ZTM4OGU4NDYxMlwvZDc2d2ZlZi0yODc4Yjc4ZC01ZGNlLTQzZjktYmUzNi0yNmVjOWJjMGRmM2IuanBnIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.98hIcOTCqXWQ67Ec5bM5eovKEn2p91mWB3uedH61ynI'\n    >>> r = requests.get(url)\n    >>> with open('grass_orig.jpg', 'bw') as f:\n    ...     f.write(r.content)\n    >>> grass_orig = imageio.imread('grass_orig.jpg')\n    >>> grass = skimage.img_as_ubyte(skimage.color.rgb2gray(grass_orig[:512, :512]))\n    >>> imageio.imwrite('grass.png', grass)\n    "
    return _load('data/grass.png', as_gray=True)