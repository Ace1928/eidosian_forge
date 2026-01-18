from yowsup.common.tools import ImageTools
from yowsup.layers.protocol_messages.protocolentities.attributes.attributes_downloadablemedia \
import os
@staticmethod
def from_filepath(filepath, dimensions=None, caption=None, jpeg_thumbnail=None):
    assert os.path.exists(filepath)
    if not jpeg_thumbnail:
        jpeg_thumbnail = ImageTools.generatePreviewFromImage(filepath)
    dimensions = dimensions or ImageTools.getImageDimensions(filepath)
    width, height = dimensions if dimensions else (None, None)
    assert width and height, 'Could not determine image dimensions, install pillow or pass dimensions'
    return ImageAttributes(DownloadableMediaMessageAttributes.from_file(filepath), width, height, caption, jpeg_thumbnail)