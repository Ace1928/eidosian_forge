import io
import pathlib
import sys
import warnings
from functools import partial
from itertools import product
import numpy as np
from ..optpkg import optional_package
import unittest
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from nibabel.arraywriters import WriterError
from nibabel.testing import (
from .. import (
from ..casting import sctypes
from ..spatialimages import SpatialImage
from ..tmpdirs import InTemporaryDirectory
from .test_api_validators import ValidateAPI
from .test_brikhead import EXAMPLE_IMAGES as AFNI_EXAMPLE_IMAGES
from .test_minc1 import EXAMPLE_IMAGES as MINC1_EXAMPLE_IMAGES
from .test_minc2 import EXAMPLE_IMAGES as MINC2_EXAMPLE_IMAGES
from .test_parrec import EXAMPLE_IMAGES as PARREC_EXAMPLE_IMAGES
def make_prox_imaker(arr, aff, hdr):

    def prox_imaker():
        img = self.image_maker(arr, aff, hdr)
        rt_img = bytesio_round_trip(img)
        return self.image_maker(rt_img.dataobj, aff, rt_img.header)
    return prox_imaker