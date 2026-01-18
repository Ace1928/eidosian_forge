import gzip
from copy import copy
from decimal import Decimal
from hashlib import sha1
from os.path import dirname
from os.path import join as pjoin
from unittest import TestCase
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ...tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from ...volumeutils import endian_codes
from .. import dicomreaders as didr
from .. import dicomwrappers as didw
from . import dicom_test, have_dicom, pydicom
def fake_frames(seq_name, field_name, value_seq):
    """Make fake frames for multiframe testing

    Parameters
    ----------
    seq_name : str
        name of sequence
    field_name : str
        name of field within sequence
    value_seq : length N sequence
        sequence of values

    Returns
    -------
    frame_seq : length N list
        each element in list is obj.<seq_name>[0].<field_name> =
        value_seq[n] for n in range(N)
    """

    class Fake:
        pass
    frames = []
    for value in value_seq:
        fake_frame = Fake()
        fake_element = Fake()
        setattr(fake_element, field_name, value)
        setattr(fake_frame, seq_name, [fake_element])
        frames.append(fake_frame)
    return frames