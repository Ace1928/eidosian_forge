from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
@lazyattr
def geotiff_tags(self):
    """Return consolidated metadata from GeoTIFF tags as dict."""
    if not self.is_geotiff:
        return
    tags = self.tags
    gkd = tags['GeoKeyDirectoryTag'].value
    if gkd[0] != 1:
        warnings.warn('invalid GeoKeyDirectoryTag')
        return {}
    result = {'KeyDirectoryVersion': gkd[0], 'KeyRevision': gkd[1], 'KeyRevisionMinor': gkd[2]}
    geokeys = TIFF.GEO_KEYS
    geocodes = TIFF.GEO_CODES
    for index in range(gkd[3]):
        keyid, tagid, count, offset = gkd[4 + index * 4:index * 4 + 8]
        keyid = geokeys.get(keyid, keyid)
        if tagid == 0:
            value = offset
        else:
            tagname = TIFF.TAGS[tagid]
            value = tags[tagname].value[offset:offset + count]
            if tagid == 34737 and count > 1 and (value[-1] == '|'):
                value = value[:-1]
            value = value if count > 1 else value[0]
        if keyid in geocodes:
            try:
                value = geocodes[keyid](value)
            except Exception:
                pass
        result[keyid] = value
    if 'IntergraphMatrixTag' in tags:
        value = tags['IntergraphMatrixTag'].value
        value = numpy.array(value)
        if len(value) == 16:
            value = value.reshape((4, 4)).tolist()
        result['IntergraphMatrix'] = value
    if 'ModelPixelScaleTag' in tags:
        value = numpy.array(tags['ModelPixelScaleTag'].value).tolist()
        result['ModelPixelScale'] = value
    if 'ModelTiepointTag' in tags:
        value = tags['ModelTiepointTag'].value
        value = numpy.array(value).reshape((-1, 6)).squeeze().tolist()
        result['ModelTiepoint'] = value
    if 'ModelTransformationTag' in tags:
        value = tags['ModelTransformationTag'].value
        value = numpy.array(value).reshape((4, 4)).tolist()
        result['ModelTransformation'] = value
    elif False:
        sx, sy, sz = tags['ModelPixelScaleTag'].value
        tiepoints = tags['ModelTiepointTag'].value
        transforms = []
        for tp in range(0, len(tiepoints), 6):
            i, j, k, x, y, z = tiepoints[tp:tp + 6]
            transforms.append([[sx, 0.0, 0.0, x - i * sx], [0.0, -sy, 0.0, y + j * sy], [0.0, 0.0, sz, z - k * sz], [0.0, 0.0, 0.0, 1.0]])
        if len(tiepoints) == 6:
            transforms = transforms[0]
        result['ModelTransformation'] = transforms
    if 'RPCCoefficientTag' in tags:
        rpcc = tags['RPCCoefficientTag'].value
        result['RPCCoefficient'] = {'ERR_BIAS': rpcc[0], 'ERR_RAND': rpcc[1], 'LINE_OFF': rpcc[2], 'SAMP_OFF': rpcc[3], 'LAT_OFF': rpcc[4], 'LONG_OFF': rpcc[5], 'HEIGHT_OFF': rpcc[6], 'LINE_SCALE': rpcc[7], 'SAMP_SCALE': rpcc[8], 'LAT_SCALE': rpcc[9], 'LONG_SCALE': rpcc[10], 'HEIGHT_SCALE': rpcc[11], 'LINE_NUM_COEFF': rpcc[12:33], 'LINE_DEN_COEFF ': rpcc[33:53], 'SAMP_NUM_COEFF': rpcc[53:73], 'SAMP_DEN_COEFF': rpcc[73:]}
    return result