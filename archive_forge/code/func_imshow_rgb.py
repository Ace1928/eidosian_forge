from types import MethodType
import numpy as np
from .axes_divider import make_axes_locatable, Size
from .mpl_axes import Axes, SimpleAxisArtist
def imshow_rgb(self, r, g, b, **kwargs):
    """
        Create the four images {rgb, r, g, b}.

        Parameters
        ----------
        r, g, b : array-like
            The red, green, and blue arrays.
        **kwargs
            Forwarded to `~.Axes.imshow` calls for the four images.

        Returns
        -------
        rgb : `~matplotlib.image.AxesImage`
        r : `~matplotlib.image.AxesImage`
        g : `~matplotlib.image.AxesImage`
        b : `~matplotlib.image.AxesImage`
        """
    if not r.shape == g.shape == b.shape:
        raise ValueError(f'Input shapes ({r.shape}, {g.shape}, {b.shape}) do not match')
    RGB = np.dstack([r, g, b])
    R = np.zeros_like(RGB)
    R[:, :, 0] = r
    G = np.zeros_like(RGB)
    G[:, :, 1] = g
    B = np.zeros_like(RGB)
    B[:, :, 2] = b
    im_rgb = self.RGB.imshow(RGB, **kwargs)
    im_r = self.R.imshow(R, **kwargs)
    im_g = self.G.imshow(G, **kwargs)
    im_b = self.B.imshow(B, **kwargs)
    return (im_rgb, im_r, im_g, im_b)