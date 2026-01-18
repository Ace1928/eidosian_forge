import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def create_colors(self):
    components = self.arrow_components()
    for component in components:
        color = self.palette.new()
        component[0].start.set_color(color)
        for arrow in component:
            arrow.set_color(color)
            arrow.end.set_color(color)