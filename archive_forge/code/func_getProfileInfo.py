from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
def getProfileInfo(profile):
    """
    (pyCMS) Gets the internal product information for the given profile.

    If ``profile`` isn't a valid CmsProfile object or filename to a profile,
    a :exc:`PyCMSError` is raised.

    If an error occurs while trying to obtain the info tag,
    a :exc:`PyCMSError` is raised.

    Use this function to obtain the information stored in the profile's
    info tag.  This often contains details about the profile, and how it
    was created, as supplied by the creator.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal profile information stored in
        an ICC tag.
    :exception PyCMSError:
    """
    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        description = profile.profile.profile_description
        cpright = profile.profile.copyright
        elements = [element for element in (description, cpright) if element]
        return '\r\n\r\n'.join(elements) + '\r\n\r\n'
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v