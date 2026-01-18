from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
def isIntentSupported(profile, intent, direction):
    """
    (pyCMS) Checks if a given intent is supported.

    Use this function to verify that you can use your desired
    ``intent`` with ``profile``, and that ``profile`` can be used for the
    input/output/proof profile as you desire.

    Some profiles are created specifically for one "direction", can cannot
    be used for others. Some profiles can only be used for certain
    rendering intents, so it's best to either verify this before trying
    to create a transform with them (using this function), or catch the
    potential :exc:`PyCMSError` that will occur if they don't
    support the modes you select.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :param intent: Integer (0-3) specifying the rendering intent you wish to
        use with this profile

            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1
            ImageCms.Intent.SATURATION            = 2
            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
            they do.
    :param direction: Integer specifying if the profile is to be used for
        input, output, or proof

            INPUT  = 0 (or use ImageCms.Direction.INPUT)
            OUTPUT = 1 (or use ImageCms.Direction.OUTPUT)
            PROOF  = 2 (or use ImageCms.Direction.PROOF)

    :returns: 1 if the intent/direction are supported, -1 if they are not.
    :exception PyCMSError:
    """
    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        if profile.profile.is_intent_supported(intent, direction):
            return 1
        else:
            return -1
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v