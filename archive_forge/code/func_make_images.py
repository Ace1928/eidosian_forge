import os
import re
import random
from gimpfu import *
def make_images(testname, pattern, alpha, layertype_in, extensions, dirname):
    assert testname.upper() == testname
    assert len(pattern) > 0
    assert len(extensions) > 0
    assert isinstance(extensions, (list, tuple))
    assert re.match('[wxtrgbcypA-F0-9]+$', pattern)
    test_alpha = 'ALPHA' in testname or 'BINARY' in testname
    grayscale = 'GRAY' in testname
    imgtype, v0_fmtinfo = {GRAY_IMAGE: (GRAY, 'BPP1G'), GRAYA_IMAGE: (GRAY, 'BPP2GA'), RGB_IMAGE: (RGB, 'BPP3'), RGBA_IMAGE: (RGB, 'BPP4'), INDEXED_IMAGE: (grayscale and GRAY or RGB, 'IX'), INDEXEDA_IMAGE: (grayscale and GRAY or RGB, 'IXA')}[layertype_in]
    PP = v0_pattern_pixel
    pixelgetter = {GRAY_IMAGE: lambda c, a: PP(c, a, 'gray'), GRAYA_IMAGE: lambda c, a: PP(c, a, 'graya'), RGB_IMAGE: lambda c, a: PP(c, a, 'rgb'), RGBA_IMAGE: lambda c, a: PP(c, a, 'rgba'), INDEXED_IMAGE: lambda c, a: PP(c, a, grayscale and 'gray' or 'rgb'), INDEXEDA_IMAGE: lambda c, a: PP(c, a, grayscale and 'graya' or 'rgba')}[layertype_in]
    layertype = {INDEXED_IMAGE: grayscale and GRAY_IMAGE or RGB_IMAGE, INDEXEDA_IMAGE: grayscale and GRAYA_IMAGE or RGBA_IMAGE}.get(layertype_in, layertype_in)
    for direction in 'xy':
        w, h = direction == 'x' and (len(pattern), 1) or (1, len(pattern))
        img = pdb.gimp_image_new(w, h, imgtype)
        lyr = pdb.gimp_layer_new(img, w, h, layertype, 'P', 100, NORMAL_MODE)
        if test_alpha:
            pdb.gimp_layer_add_alpha(lyr)
            pdb.gimp_drawable_fill(lyr, TRANSPARENT_FILL)
        pdb.gimp_image_add_layer(img, lyr, 0)
        draw_pattern(lyr, pattern, alpha, direction, pixelgetter)
        if layertype_in in (INDEXED_IMAGE, INDEXEDA_IMAGE):
            colors = len(set(pattern)) + (test_alpha and 1 or 0)
            pdb.gimp_convert_indexed(img, 0, 0, colors, 0, 0, 'ignored')
        for ext in extensions:
            save_image(dirname, img, lyr, w, h, pattern, alpha, v0_fmtinfo, testname, ext)