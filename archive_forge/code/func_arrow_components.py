import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def arrow_components(self, include_isolated_vertices=False, distinguish_closed=False):
    """
        Returns a list of components, given as lists of arrows.
        The closed components are sorted in DT order if they have
        been marked.  The others are sorted by age. If distinguish_closed
        is set to True then two lists are returned, the first has the closed
        components the second has the non-closed components.
        """
    pool = [v.out_arrow for v in self.Vertices if v.in_arrow is None]
    pool += [v.out_arrow for v in self.Vertices if v.in_arrow is not None]
    closed, nonclosed = ([], [])
    while pool:
        first_arrow = pool.pop(0)
        if first_arrow is None:
            continue
        component = [first_arrow]
        while component[-1].end is not component[0].start:
            next_arrow = component[-1].end.out_arrow
            if next_arrow is None:
                break
            pool.remove(next_arrow)
            component.append(next_arrow)
        if next_arrow is None:
            nonclosed.append(component)
        else:
            closed.append(component)
    if include_isolated_vertices:
        for vertex in [v for v in self.Vertices if v.is_isolated()]:
            nonclosed.append([Arrow(vertex, vertex, self.canvas, color=vertex.color)])

    def oldest_vertex(component):

        def oldest(arrow):
            return min([self.Vertices.index(v) for v in [arrow.start, arrow.end] if v])
        return min([len(self.Vertices)] + [oldest(a) for a in component])
    closed.sort(key=lambda x: (x[0].component, oldest_vertex(x)))
    nonclosed.sort(key=oldest_vertex)
    return (closed, nonclosed) if distinguish_closed else closed + nonclosed