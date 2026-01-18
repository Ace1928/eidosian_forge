import os
import string
import struct
import warnings
import numpy as np
import nibabel as nib
from nibabel.openers import Opener
from nibabel.orientations import aff2axcodes, axcodes2ornt
from nibabel.volumeutils import endian_codes, native_code, swapped_code
from .array_sequence import create_arraysequences_from_generator
from .header import Field
from .tractogram import LazyTractogram, Tractogram, TractogramItem
from .tractogram_file import DataError, HeaderError, HeaderWarning, TractogramFile
from .utils import peek_next
def get_affine_trackvis_to_rasmm(header):
    """Get affine mapping trackvis voxelmm space to RAS+ mm space

    The streamlines in a trackvis file are in 'voxelmm' space, where the
    coordinates refer to the corner of the voxel.

    Compute the affine matrix that will bring them back to RAS+ mm space, where
    the coordinates refer to the center of the voxel.

    Parameters
    ----------
    header : dict or ndarray
        Dict or numpy structured array containing trackvis header.

    Returns
    -------
    aff_tv2ras : shape (4, 4) array
        Affine array mapping coordinates in 'voxelmm' space to RAS+ mm space.
    """
    affine = np.eye(4)
    scale = np.eye(4)
    scale[range(3), range(3)] /= header[Field.VOXEL_SIZES]
    affine = np.dot(scale, affine)
    offset = np.eye(4)
    offset[:-1, -1] -= 0.5
    affine = np.dot(offset, affine)
    vox_order = header[Field.VOXEL_ORDER]
    if hasattr(vox_order, 'item'):
        vox_order = header[Field.VOXEL_ORDER].item()
    affine_ornt = ''.join(aff2axcodes(header[Field.VOXEL_TO_RASMM]))
    header_ornt = axcodes2ornt(vox_order.decode('latin1').upper())
    affine_ornt = axcodes2ornt(affine_ornt)
    ornt = nib.orientations.ornt_transform(header_ornt, affine_ornt)
    M = nib.orientations.inv_ornt_aff(ornt, header[Field.DIMENSIONS])
    affine = np.dot(M, affine)
    voxel_to_rasmm = header[Field.VOXEL_TO_RASMM]
    affine_voxmm_to_rasmm = np.dot(voxel_to_rasmm, affine)
    return affine_voxmm_to_rasmm.astype(np.float32)