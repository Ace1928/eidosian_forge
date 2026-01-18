from .base import Renderer
def draw_image(self, imdata, extent, coordinates, style, mplobj=None):
    self.output += '    draw image of size {0}\n'.format(len(imdata))