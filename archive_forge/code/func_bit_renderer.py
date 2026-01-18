from OpenGL.EGL import *
import itertools
def bit_renderer(bit):

    def render(value):
        if bit.name in value:
            return ' Y'
        else:
            return ' .'
    return render