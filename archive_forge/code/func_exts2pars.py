import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
from locale import getpreferredencoding
import numpy as np
from .affines import apply_affine, dot_reduce, from_matvec
from .eulerangles import euler2mat
from .fileslice import fileslice, strided_scalar
from .nifti1 import unit_codes
from .openers import ImageOpener
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import Recoder, array_from_file
def exts2pars(exts_source):
    """Parse, return any PAR headers from NIfTI extensions in `exts_source`

    Parameters
    ----------
    exts_source : sequence or `Nifti1Image`, `Nifti1Header` instance
        A sequence of extensions, or header containing NIfTI extensions, or an
        image containing a header with NIfTI extensions.

    Returns
    -------
    par_headers : list
        A list of PARRECHeader objects, usually empty or with one element, each
        element contains a PARRECHeader read from the contained extensions.
    """
    headers = []
    exts_source = exts_source.header if hasattr(exts_source, 'header') else exts_source
    exts_source = exts_source.extensions if hasattr(exts_source, 'extensions') else exts_source
    for extension in exts_source:
        content = extension.get_content()
        content = content.decode(getpreferredencoding(False))
        if not content.startswith('# === DATA DESCRIPTION FILE ==='):
            continue
        gen_info, image_info = parse_PAR_header(StringIO(content))
        headers.append(PARRECHeader(gen_info, image_info))
    return headers