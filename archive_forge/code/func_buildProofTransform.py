from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
def buildProofTransform(inputProfile, outputProfile, proofProfile, inMode, outMode, renderingIntent=Intent.PERCEPTUAL, proofRenderingIntent=Intent.ABSOLUTE_COLORIMETRIC, flags=FLAGS['SOFTPROOFING']):
    """
    (pyCMS) Builds an ICC transform mapping from the ``inputProfile`` to the
    ``outputProfile``, but tries to simulate the result that would be
    obtained on the ``proofProfile`` device.

    If the input, output, or proof profiles specified are not valid
    filenames, a :exc:`PyCMSError` will be raised.

    If an error occurs during creation of the transform,
    a :exc:`PyCMSError` will be raised.

    If ``inMode`` or ``outMode`` are not a mode supported by the ``outputProfile``
    (or by pyCMS), a :exc:`PyCMSError` will be raised.

    This function builds and returns an ICC transform from the ``inputProfile``
    to the ``outputProfile``, but tries to simulate the result that would be
    obtained on the ``proofProfile`` device using ``renderingIntent`` and
    ``proofRenderingIntent`` to determine what to do with out-of-gamut
    colors.  This is known as "soft-proofing".  It will ONLY work for
    converting images that are in ``inMode`` to images that are in outMode
    color format (PIL mode, i.e. "RGB", "RGBA", "CMYK", etc.).

    Usage of the resulting transform object is exactly the same as with
    ImageCms.buildTransform().

    Proof profiling is generally used when using an output device to get a
    good idea of what the final printed/displayed image would look like on
    the ``proofProfile`` device when it's quicker and easier to use the
    output device for judging color.  Generally, this means that the
    output device is a monitor, or a dye-sub printer (etc.), and the simulated
    device is something more expensive, complicated, or time consuming
    (making it difficult to make a real print for color judgement purposes).

    Soft-proofing basically functions by adjusting the colors on the
    output device to match the colors of the device being simulated. However,
    when the simulated device has a much wider gamut than the output
    device, you may obtain marginal results.

    :param inputProfile: String, as a valid filename path to the ICC input
        profile you wish to use for this transform, or a profile object
    :param outputProfile: String, as a valid filename path to the ICC output
        (monitor, usually) profile you wish to use for this transform, or a
        profile object
    :param proofProfile: String, as a valid filename path to the ICC proof
        profile you wish to use for this transform, or a profile object
    :param inMode: String, as a valid PIL mode that the appropriate profile
        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)
    :param outMode: String, as a valid PIL mode that the appropriate profile
        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)
    :param renderingIntent: Integer (0-3) specifying the rendering intent you
        wish to use for the input->proof (simulated) transform

            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1
            ImageCms.Intent.SATURATION            = 2
            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
        they do.
    :param proofRenderingIntent: Integer (0-3) specifying the rendering intent
        you wish to use for proof->output transform

            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1
            ImageCms.Intent.SATURATION            = 2
            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
        they do.
    :param flags: Integer (0-...) specifying additional flags
    :returns: A CmsTransform class object.
    :exception PyCMSError:
    """
    if not isinstance(renderingIntent, int) or not 0 <= renderingIntent <= 3:
        msg = 'renderingIntent must be an integer between 0 and 3'
        raise PyCMSError(msg)
    if not isinstance(flags, int) or not 0 <= flags <= _MAX_FLAG:
        msg = 'flags must be an integer between 0 and %s' + _MAX_FLAG
        raise PyCMSError(msg)
    try:
        if not isinstance(inputProfile, ImageCmsProfile):
            inputProfile = ImageCmsProfile(inputProfile)
        if not isinstance(outputProfile, ImageCmsProfile):
            outputProfile = ImageCmsProfile(outputProfile)
        if not isinstance(proofProfile, ImageCmsProfile):
            proofProfile = ImageCmsProfile(proofProfile)
        return ImageCmsTransform(inputProfile, outputProfile, inMode, outMode, renderingIntent, proofProfile, proofRenderingIntent, flags)
    except (OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v