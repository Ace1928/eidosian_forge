from __future__ import annotations
import os
import struct
import sys
from . import Image, ImageFile
def loadImageSeries(filelist=None):
    """create a list of :py:class:`~PIL.Image.Image` objects for use in a montage"""
    if filelist is None or len(filelist) < 1:
        return
    imglist = []
    for img in filelist:
        if not os.path.exists(img):
            print(f'unable to find {img}')
            continue
        try:
            with Image.open(img) as im:
                im = im.convert2byte()
        except Exception:
            if not isSpiderImage(img):
                print(img + ' is not a Spider image file')
            continue
        im.info['filename'] = img
        imglist.append(im)
    return imglist