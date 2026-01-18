import os.path as op
import nibabel as nb
import numpy as np
from nibabel.volumeutils import native_code
from nibabel.orientations import aff2axcodes
from ... import logging
from ...utils.filemanip import split_filename
from ..base import TraitedSpec, File, isdefined
from ..dipy.base import DipyBaseInterface, HAVE_DIPY as have_dipy
def read_mrtrix_tracks(in_file, as_generator=True):
    header = read_mrtrix_header(in_file)
    streamlines = read_mrtrix_streamlines(in_file, header, as_generator)
    return (header, streamlines)