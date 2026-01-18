import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def oldest_vertex(component):

    def oldest(arrow):
        return min([self.Vertices.index(v) for v in [arrow.start, arrow.end] if v])
    return min([len(self.Vertices)] + [oldest(a) for a in component])