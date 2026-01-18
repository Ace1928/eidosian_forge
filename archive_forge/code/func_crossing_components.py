import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def crossing_components(self):
    """
        Returns a list of lists of ECrossings, one per component,
        where the corresponding crossings are ordered consecutively
        through the component.  Requires that all components be closed.
        """
    for vertex in self.Vertices:
        if vertex.is_endpoint():
            raise ValueError('All components must be closed.')
    result = []
    arrow_components = self.arrow_components()
    for component in arrow_components:
        crosses = []
        for arrow in component:
            arrow_crosses = [(c.height(arrow), c, arrow) for c in self.Crossings if arrow in c]
            arrow_crosses.sort()
            crosses += arrow_crosses
        result.append([ECrossing(c[1], c[2]) for c in crosses])
    for crossing in self.Crossings:
        crossing.clear_marks()
    for component in result:
        for ecrossing in component:
            ecrossing.crossing.mark_component(component)
    return result