import pyglet
from pyglet.gl import *
def get_max_color_attachments():
    """Get the maximum allow Framebuffer Color attachements"""
    number = GLint()
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, number)
    return number.value