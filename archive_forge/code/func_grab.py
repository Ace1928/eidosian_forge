from __future__ import annotations
import io
import os
import shutil
import subprocess
import sys
import tempfile
from . import Image
def grab(bbox=None, include_layered_windows=False, all_screens=False, xdisplay=None):
    if xdisplay is None:
        if sys.platform == 'darwin':
            fh, filepath = tempfile.mkstemp('.png')
            os.close(fh)
            args = ['screencapture']
            if bbox:
                left, top, right, bottom = bbox
                args += ['-R', f'{left},{top},{right - left},{bottom - top}']
            subprocess.call(args + ['-x', filepath])
            im = Image.open(filepath)
            im.load()
            os.unlink(filepath)
            if bbox:
                im_resized = im.resize((right - left, bottom - top))
                im.close()
                return im_resized
            return im
        elif sys.platform == 'win32':
            offset, size, data = Image.core.grabscreen_win32(include_layered_windows, all_screens)
            im = Image.frombytes('RGB', size, data, 'raw', 'BGR', size[0] * 3 + 3 & -4, -1)
            if bbox:
                x0, y0 = offset
                left, top, right, bottom = bbox
                im = im.crop((left - x0, top - y0, right - x0, bottom - y0))
            return im
    try:
        if not Image.core.HAVE_XCB:
            msg = 'Pillow was built without XCB support'
            raise OSError(msg)
        size, data = Image.core.grabscreen_x11(xdisplay)
    except OSError:
        if xdisplay is None and sys.platform not in ('darwin', 'win32') and shutil.which('gnome-screenshot'):
            fh, filepath = tempfile.mkstemp('.png')
            os.close(fh)
            subprocess.call(['gnome-screenshot', '-f', filepath])
            im = Image.open(filepath)
            im.load()
            os.unlink(filepath)
            if bbox:
                im_cropped = im.crop(bbox)
                im.close()
                return im_cropped
            return im
        else:
            raise
    else:
        im = Image.frombytes('RGB', size, data, 'raw', 'BGRX', size[0] * 4, 1)
        if bbox:
            im = im.crop(bbox)
        return im