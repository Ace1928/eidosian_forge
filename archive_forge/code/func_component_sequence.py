import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def component_sequence(component):
    sequence = []
    for arrow in component:
        this_arrows_crossings = []
        for index, virtual_crossing in enumerate(virtual_crossings):
            if arrow == virtual_crossing.under:
                other_arrow = virtual_crossing.over
            elif arrow == virtual_crossing.over:
                other_arrow = virtual_crossing.under
            else:
                continue
            sign = arrow.dx * other_arrow.dy - arrow.dy * other_arrow.dx > 0
            this_arrows_crossings.append((arrow ^ other_arrow, index, '+' if sign else '-'))
        this_arrows_crossings.sort()
        sequence += [pm + str(index) for _, index, pm in this_arrows_crossings]
    return sequence