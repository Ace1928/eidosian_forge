import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def saturate_face(face_info):
    for i, a in enumerate(face_info):
        if a.turn == -1:
            face_info = face_info[i:] + face_info[:i]
            break
    for i in range(len(face_info) - 2):
        x, y, z = face_info[i:i + 3]
        if x.turn == -1 and y.turn == z.turn == 1:
            a, b = (x, z) if x.kind == 'sink' else (z, x)
            remaining = face_info[:i] + [LabeledFaceVertex(z.index, z.kind, 1)] + face_info[i + 3:]
            return [(a.index, b.index)] + saturate_face(remaining)
    return []