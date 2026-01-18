from functools import reduce
import logging
import operator
import click
from . import options
import rasterio
from rasterio.enums import _OverviewResampling as OverviewResampling
def get_maximum_overview_level(width, height, minsize=256):
    """
    Calculate the maximum overview level of a dataset at which
    the smallest overview is smaller than `minsize`.

    Attributes
    ----------
    width : int
        Width of the dataset.
    height : int
        Height of the dataset.
    minsize : int (default: 256)
        Minimum overview size.

    Returns
    -------
    overview_level: int
        overview level.

    """
    overview_level = 0
    overview_factor = 1
    while min(width // overview_factor, height // overview_factor) > minsize:
        overview_factor *= 2
        overview_level += 1
    return overview_level