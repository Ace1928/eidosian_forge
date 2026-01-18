from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def compute_tilts(self):
    """
        Computes all tilts. They are written to the instances of
        t3m.simplex.Face and can be accessed as
        [ face.Tilt for face in crossSection.Faces].
        """
    for face in self.mcomplex.Faces:
        face.Tilt = RealCuspCrossSection._face_tilt(face)