from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
def getProfileName(profile):
    """

    (pyCMS) Gets the internal product name for the given profile.

    If ``profile`` isn't a valid CmsProfile object or filename to a profile,
    a :exc:`PyCMSError` is raised If an error occurs while trying
    to obtain the name tag, a :exc:`PyCMSError` is raised.

    Use this function to obtain the INTERNAL name of the profile (stored
    in an ICC tag in the profile itself), usually the one used when the
    profile was originally created.  Sometimes this tag also contains
    additional information supplied by the creator.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal name of the profile as stored
        in an ICC tag.
    :exception PyCMSError:
    """
    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        model = profile.profile.model
        manufacturer = profile.profile.manufacturer
        if not (model or manufacturer):
            return (profile.profile.profile_description or '') + '\n'
        if not manufacturer or len(model) > 30:
            return model + '\n'
        return f'{model} - {manufacturer}\n'
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v