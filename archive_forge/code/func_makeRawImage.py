import os
import binascii
from io import BytesIO
from reportlab import rl_config
from reportlab.lib.utils import ImageReader, isUnicode
from reportlab.lib.rl_accel import asciiBase85Encode, asciiBase85Decode
def makeRawImage(filename, IMG=None, detectJpeg=False):
    import zlib
    img = ImageReader(filename)
    if IMG is not None:
        IMG.append(img)
        if detectJpeg and img.jpeg_fh():
            return None
    imgwidth, imgheight = img.getSize()
    raw = img.getRGBData()
    code = []
    append = code.append
    append('BI')
    append('/W %s /H %s /BPC 8 /CS /%s /F [/Fl]' % (imgwidth, imgheight, _mode2cs[img.mode]))
    append('ID')
    assert len(raw) == imgwidth * imgheight * _mode2bpp[img.mode], 'Wrong amount of data for image'
    compressed = zlib.compress(raw)
    _chunker(compressed, code)
    append('EI')
    return code