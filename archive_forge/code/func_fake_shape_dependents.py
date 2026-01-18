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
def fake_shape_dependents(div_seq, sid_seq=None, sid_dim=None):
    """Make a fake dictionary of data that ``image_shape`` is dependent on.

    Parameters
    ----------
    div_seq : list of tuples
        list of values to use for the `DimensionIndexValues` of each frame.
    sid_seq : list of int
        list of values to use for the `StackID` of each frame.
    sid_dim : int
        the index of the column in 'div_seq' to use as 'sid_seq'
    """

    class DimIdxSeqElem:

        def __init__(self, dip=(0, 0), fgp=None):
            self.DimensionIndexPointer = dip
            if fgp is not None:
                self.FunctionalGroupPointer = fgp

    class FrmContSeqElem:

        def __init__(self, div, sid):
            self.DimensionIndexValues = div
            self.StackID = sid

    class PerFrmFuncGrpSeqElem:

        def __init__(self, div, sid):
            self.FrameContentSequence = [FrmContSeqElem(div, sid)]
    if sid_seq is None:
        if sid_dim is None:
            sid_dim = 0
        sid_seq = [div[sid_dim] for div in div_seq]
    num_of_frames = len(div_seq)
    dim_idx_seq = [DimIdxSeqElem()] * num_of_frames
    if sid_dim is not None:
        sid_tag = pydicom.datadict.tag_for_keyword('StackID')
        fcs_tag = pydicom.datadict.tag_for_keyword('FrameContentSequence')
        dim_idx_seq[sid_dim] = DimIdxSeqElem(sid_tag, fcs_tag)
    frames = [PerFrmFuncGrpSeqElem(div, sid) for div, sid in zip(div_seq, sid_seq)]
    return {'NumberOfFrames': num_of_frames, 'DimensionIndexSequence': dim_idx_seq, 'PerFrameFunctionalGroupsSequence': frames}