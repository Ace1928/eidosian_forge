from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
def createProfile(colorSpace, colorTemp=-1):
    """
    (pyCMS) Creates a profile.

    If colorSpace not in ``["LAB", "XYZ", "sRGB"]``,
    a :exc:`PyCMSError` is raised.

    If using LAB and ``colorTemp`` is not a positive integer,
    a :exc:`PyCMSError` is raised.

    If an error occurs while creating the profile,
    a :exc:`PyCMSError` is raised.

    Use this function to create common profiles on-the-fly instead of
    having to supply a profile on disk and knowing the path to it.  It
    returns a normal CmsProfile object that can be passed to
    ImageCms.buildTransformFromOpenProfiles() to create a transform to apply
    to images.

    :param colorSpace: String, the color space of the profile you wish to
        create.
        Currently only "LAB", "XYZ", and "sRGB" are supported.
    :param colorTemp: Positive integer for the white point for the profile, in
        degrees Kelvin (i.e. 5000, 6500, 9600, etc.).  The default is for D50
        illuminant if omitted (5000k).  colorTemp is ONLY applied to LAB
        profiles, and is ignored for XYZ and sRGB.
    :returns: A CmsProfile class object
    :exception PyCMSError:
    """
    if colorSpace not in ['LAB', 'XYZ', 'sRGB']:
        msg = f'Color space not supported for on-the-fly profile creation ({colorSpace})'
        raise PyCMSError(msg)
    if colorSpace == 'LAB':
        try:
            colorTemp = float(colorTemp)
        except (TypeError, ValueError) as e:
            msg = f'Color temperature must be numeric, "{colorTemp}" not valid'
            raise PyCMSError(msg) from e
    try:
        return core.createProfile(colorSpace, colorTemp)
    except (TypeError, ValueError) as v:
        raise PyCMSError(v) from v