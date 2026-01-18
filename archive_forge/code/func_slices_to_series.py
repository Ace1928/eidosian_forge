import glob
from os.path import join as pjoin
import numpy as np
from .. import Nifti1Image
from .dicomwrappers import wrapper_from_data, wrapper_from_file
def slices_to_series(wrappers):
    """Sort sequence of slice wrappers into series

    This follows the SPM model fairly closely

    Parameters
    ----------
    wrappers : sequence
       sequence of ``Wrapper`` objects for sorting into volumes

    Returns
    -------
    series : sequence
       sequence of sequences of wrapper objects, where each sequence is
       wrapper objects comprising a series, sorted into slice order
    """
    volume_lists = [wrappers[0:1]]
    for dw in wrappers[1:]:
        for vol_list in volume_lists:
            if dw.is_same_series(vol_list[0]):
                vol_list.append(dw)
                break
        else:
            volume_lists.append([dw])
    print('We appear to have %d Series' % len(volume_lists))
    out_vol_lists = []
    for vol_list in volume_lists:
        if len(vol_list) > 1:
            vol_list.sort(key=_slice_sorter)
            zs = [s.slice_indicator for s in vol_list]
            if len(set(zs)) < len(zs):
                out_vol_lists += _third_pass(vol_list)
                continue
        out_vol_lists.append(vol_list)
    print('We have %d volumes after second pass' % len(out_vol_lists))
    for vol_list in out_vol_lists:
        zs = [s.slice_indicator for s in vol_list]
        diffs = np.diff(zs)
        if not np.allclose(diffs, np.mean(diffs)):
            raise DicomReadError('Largeish slice gaps - missing DICOMs?')
    return out_vol_lists