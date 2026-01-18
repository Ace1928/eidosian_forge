import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def test_face_method(N):
    for i in range(N):
        check_faces(random_link())