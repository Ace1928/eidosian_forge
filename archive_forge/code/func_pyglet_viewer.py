import os
from os.path import join
import shutil
import tempfile
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.misc import debug
from .latex import latex
def pyglet_viewer(fname, fmt):
    try:
        from pyglet import window, image, gl
        from pyglet.window import key
        from pyglet.image.codecs import ImageDecodeException
    except ImportError:
        raise ImportError('pyglet is required for preview.\n visit https://pyglet.org/')
    try:
        img = image.load(fname)
    except ImageDecodeException:
        raise ValueError("pyglet preview does not work for '{}' files.".format(fmt))
    offset = 25
    config = gl.Config(double_buffer=False)
    win = window.Window(width=img.width + 2 * offset, height=img.height + 2 * offset, caption='SymPy', resizable=False, config=config)
    win.set_vsync(False)
    try:

        def on_close():
            win.has_exit = True
        win.on_close = on_close

        def on_key_press(symbol, modifiers):
            if symbol in [key.Q, key.ESCAPE]:
                on_close()
        win.on_key_press = on_key_press

        def on_expose():
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            img.blit((win.width - img.width) / 2, (win.height - img.height) / 2)
        win.on_expose = on_expose
        while not win.has_exit:
            win.dispatch_events()
            win.flip()
    except KeyboardInterrupt:
        pass
    win.close()