import argparse
import itertools
import sys
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import VideoTemplate_generic as ui_template
def mkData():
    with pg.BusyCursor():
        global data, cache, ui, xp
        frames = ui.framesSpin.value()
        width = ui.widthSpin.value()
        height = ui.heightSpin.value()
        cacheKey = (ui.dtypeCombo.currentText(), ui.rgbCheck.isChecked(), frames, width, height)
        if cacheKey not in cache:
            if cacheKey[0] == 'uint8':
                dt = xp.uint8
                loc = 128
                scale = 64
                mx = 255
            elif cacheKey[0] == 'uint16':
                dt = xp.uint16
                loc = 4096
                scale = 1024
                mx = 2 ** 16 - 1
            elif cacheKey[0] == 'float':
                dt = xp.float32
                loc = 1.0
                scale = 0.1
                mx = 1.0
            else:
                raise ValueError(f'unable to handle dtype: {cacheKey[0]}')
            chan_shape = (height, width)
            if ui.rgbCheck.isChecked():
                frame_shape = chan_shape + (3,)
            else:
                frame_shape = chan_shape
            data = xp.empty((frames,) + frame_shape, dtype=dt)
            view = data.reshape((-1,) + chan_shape)
            for idx in range(view.shape[0]):
                subdata = xp.random.normal(loc=loc, scale=scale, size=chan_shape)
                if cacheKey[0] != 'float':
                    xp.clip(subdata, 0, mx, out=subdata)
                view[idx] = subdata
            data[:, 10:50, 10] = mx
            data[:, 48, 9:12] = mx
            data[:, 47, 8:13] = mx
            cache = {cacheKey: data}
        data = cache[cacheKey]
        updateLUT()
        updateSize()