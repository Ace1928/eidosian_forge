from __future__ import annotations
from . import Image, ImageFilter, ImageStat
def enhance(self, factor):
    """
        Returns an enhanced image.

        :param factor: A floating point value controlling the enhancement.
                       Factor 1.0 always returns a copy of the original image,
                       lower factors mean less color (brightness, contrast,
                       etc), and higher values more. There are no restrictions
                       on this value.
        :rtype: :py:class:`~PIL.Image.Image`
        """
    return Image.blend(self.degenerate, self.image, factor)