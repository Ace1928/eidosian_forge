import sys
from distutils.core import Distribution
import warnings
import distutils.core
import distutils.dist
from numpy.distutils.extension import Extension  # noqa: F401
from numpy.distutils.numpy_distribution import NumpyDistribution
from numpy.distutils.command import config, config_compiler, \
from numpy.distutils.misc_util import is_sequence, is_string
def get_distribution(always=False):
    dist = distutils.core._setup_distribution
    if dist is not None and 'DistributionWithoutHelpCommands' in repr(dist):
        dist = None
    if always and dist is None:
        dist = NumpyDistribution()
    return dist