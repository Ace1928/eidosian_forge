import pickle
import base64
import zlib
from re import match as re_match
from collections import deque
from math import sqrt, pi, radians, acos, atan, atan2, pow, floor
from math import sin as math_sin, cos as math_cos
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.properties import ListProperty
from kivy.compat import PY2
from io import BytesIO
def match_candidate(self, cand, **kwargs):
    """Match a given candidate against this MultistrokeGesture object. Will
        test against all templates and report results as a list of four
        items:

            `index 0`
                Best matching template's index (in self.templates)
            `index 1`
                Computed distance from the template to the candidate path
            `index 2`
                List of distances for all templates. The list index
                corresponds to a :class:`UnistrokeTemplate` index in
                self.templates.
            `index 3`
                Counter for the number of performed matching operations, ie
                templates matched against the candidate
        """
    best_d = float('infinity')
    best_tpl = None
    mos = 0
    out = []
    if self.stroke_sens and len(self.strokes) != len(cand.strokes):
        return (best_tpl, best_d, out, mos)
    skip_bounded = cand.skip_bounded
    skip_invariant = cand.skip_invariant
    get_distance = self.get_distance
    ang_sim_threshold = self.angle_similarity_threshold()
    for idx, tpl in enumerate(self.templates):
        if tpl.orientation_sens:
            if skip_bounded:
                continue
        elif skip_invariant:
            continue
        mos += 1
        n = kwargs.get('force_numpoints', tpl.numpoints)
        ang_sim = cand.get_angle_similarity(tpl, numpoints=n)
        if ang_sim > ang_sim_threshold:
            continue
        d = get_distance(cand, tpl, numpoints=n)
        out.append(d)
        if d < best_d:
            best_d = d
            best_tpl = idx
    return (best_tpl, best_d, out, mos)