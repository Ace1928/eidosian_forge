from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
def getProfileDescription(profile):
    """
    (pyCMS) Gets the description for the given profile.

    If ``profile`` isn't a valid CmsProfile object or filename to a profile, a
    :exc:`PyCMSError` is raised.

    If an error occurs while trying to obtain the description tag,
    a :exc:`PyCMSError` is raised.

    Use this function to obtain the information stored in the profile's
    description tag.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal profile information stored in an
        ICC tag.
    :exception PyCMSError:
    """
    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        return (profile.profile.profile_description or '') + '\n'
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v