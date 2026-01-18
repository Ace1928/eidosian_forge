import glob
import numpy as np
import numpy.linalg as npl
import nibabel as nib
from nibabel import parrec
from nibabel.affines import to_matvec
from nibabel.optpkg import optional_package
def resample_img2img(img_to, img_from, order=1, out_class=nib.Nifti1Image):
    if not have_scipy:
        raise Exception('Scipy must be installed to run resample_img2img.')
    from scipy import ndimage as spnd
    vox2vox = npl.inv(img_from.affine).dot(img_to.affine)
    rzs, trans = to_matvec(vox2vox)
    data = spnd.affine_transform(img_from.get_fdata(), rzs, trans, img_to.shape, order=order)
    return out_class(data, img_to.affine)