from __future__ import annotations
import collections
import os
import sys
import warnings
import PIL
from . import Image
def pilinfo(out=None, supported_formats=True):
    """
    Prints information about this installation of Pillow.
    This function can be called with ``python3 -m PIL``.

    :param out:
        The output stream to print to. Defaults to ``sys.stdout`` if ``None``.
    :param supported_formats:
        If ``True``, a list of all supported image file formats will be printed.
    """
    if out is None:
        out = sys.stdout
    Image.init()
    print('-' * 68, file=out)
    print(f'Pillow {PIL.__version__}', file=out)
    py_version = sys.version.splitlines()
    print(f'Python {py_version[0].strip()}', file=out)
    for py_version in py_version[1:]:
        print(f'       {py_version.strip()}', file=out)
    print('-' * 68, file=out)
    print(f'Python modules loaded from {os.path.dirname(Image.__file__)}', file=out)
    print(f'Binary modules loaded from {os.path.dirname(Image.core.__file__)}', file=out)
    print('-' * 68, file=out)
    for name, feature in [('pil', 'PIL CORE'), ('tkinter', 'TKINTER'), ('freetype2', 'FREETYPE2'), ('littlecms2', 'LITTLECMS2'), ('webp', 'WEBP'), ('transp_webp', 'WEBP Transparency'), ('webp_mux', 'WEBPMUX'), ('webp_anim', 'WEBP Animation'), ('jpg', 'JPEG'), ('jpg_2000', 'OPENJPEG (JPEG2000)'), ('zlib', 'ZLIB (PNG/ZIP)'), ('libtiff', 'LIBTIFF'), ('raqm', 'RAQM (Bidirectional Text)'), ('libimagequant', 'LIBIMAGEQUANT (Quantization method)'), ('xcb', 'XCB (X protocol)')]:
        if check(name):
            if name == 'jpg' and check_feature('libjpeg_turbo'):
                v = 'libjpeg-turbo ' + version_feature('libjpeg_turbo')
            else:
                v = version(name)
            if v is not None:
                version_static = name in ('pil', 'jpg')
                if name == 'littlecms2':
                    version_static = tuple((int(x) for x in v.split('.'))) < (2, 7)
                t = 'compiled for' if version_static else 'loaded'
                if name == 'raqm':
                    for f in ('fribidi', 'harfbuzz'):
                        v2 = version_feature(f)
                        if v2 is not None:
                            v += f', {f} {v2}'
                print('---', feature, 'support ok,', t, v, file=out)
            else:
                print('---', feature, 'support ok', file=out)
        else:
            print('***', feature, 'support not installed', file=out)
    print('-' * 68, file=out)
    if supported_formats:
        extensions = collections.defaultdict(list)
        for ext, i in Image.EXTENSION.items():
            extensions[i].append(ext)
        for i in sorted(Image.ID):
            line = f'{i}'
            if i in Image.MIME:
                line = f'{line} {Image.MIME[i]}'
            print(line, file=out)
            if i in extensions:
                print('Extensions: {}'.format(', '.join(sorted(extensions[i]))), file=out)
            features = []
            if i in Image.OPEN:
                features.append('open')
            if i in Image.SAVE:
                features.append('save')
            if i in Image.SAVE_ALL:
                features.append('save_all')
            if i in Image.DECODERS:
                features.append('decode')
            if i in Image.ENCODERS:
                features.append('encode')
            print('Features: {}'.format(', '.join(features)), file=out)
            print('-' * 68, file=out)