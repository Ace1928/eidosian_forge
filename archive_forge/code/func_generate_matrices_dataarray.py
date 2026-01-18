import numpy as np
import xarray as xr
def generate_matrices_dataarray(seed=None):
    """Generate a 4d DataArray representing a batch of matrices.

    Parameters
    ----------
    seed : int or sequence of int, optional
        The random seed used to initialize :func:`numpy.random.default_rng`.

    Examples
    --------
    The dataset generated is the following:

    .. jupyter-execute::

        from xarray_einstats import tutorial
        tutorial.generate_matrices_dataarray(5)

    Notes
    -----
    This function is not part of the public API and is designed for use in our documentation.
    In addition to generating the data, it also sets ``display_expand_data=False`` to
    avoid taking too much virtual space with examples.

    """
    xr.set_options(display_expand_data=False)
    rng = np.random.default_rng(seed)
    return xr.DataArray(rng.exponential(size=(10, 3, 4, 4)), dims=['batch', 'experiment', 'dim', 'dim2'])