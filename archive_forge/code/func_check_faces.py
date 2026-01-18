import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def check_faces(link):
    faces = link.faces()
    assert len(link.vertices) - len(link.edges) + len(faces) == 2
    assert all((val == 1 for val in Counter(sum(faces, [])).values()))
    assert link.is_planar()