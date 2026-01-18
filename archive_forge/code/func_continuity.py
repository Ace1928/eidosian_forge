from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
@array
def continuity(detections, annotations, phase_tolerance=CONTINUITY_PHASE_TOLERANCE, tempo_tolerance=CONTINUITY_TEMPO_TOLERANCE, offbeat=True, double=True, triple=True):
    """
    Calculate the cmlc, cmlt, amlc and amlt scores for the given detections and
    annotations.

    Parameters
    ----------
    detections : list or numpy array
        Detected beats.
    annotations : list or numpy array
        Annotated beats.
    phase_tolerance : float, optional
        Allowed phase tolerance.
    tempo_tolerance : float, optional
        Allowed tempo tolerance.
    offbeat : bool, optional
        Include offbeat variation.
    double  : bool, optional
        Include double and half tempo variations (and offbeat thereof).
    triple  : bool, optional
        Include triple and third tempo variations (and offbeats thereof).

    Returns
    -------
    cmlc : float
        Tracking accuracy, continuity at the correct metrical level required.
    cmlt : float
        Same as cmlc, continuity at the correct metrical level not required.
    amlc : float
        Same as cmlc, alternate metrical levels allowed.
    amlt : float
        Same as cmlt, alternate metrical levels allowed.

    See Also
    --------
    :func:`cml`

    """
    if len(detections) == 0 and len(annotations) == 0:
        return (1.0, 1.0, 1.0, 1.0)
    if len(detections) <= 1 or len(annotations) <= 1:
        return (0.0, 0.0, 0.0, 0.0)
    cmlc, cmlt = cml(detections, annotations, tempo_tolerance, phase_tolerance)
    amlc = cmlc
    amlt = cmlt
    if cmlc > 0.5:
        return (cmlc, cmlt, amlc, amlt)
    sequences = variations(annotations, offbeat=offbeat, double=double, half=double, triple=triple, third=triple)
    for sequence in sequences:
        try:
            c, t = cml(detections, sequence, tempo_tolerance, phase_tolerance)
        except BeatIntervalError:
            c, t = (np.nan, np.nan)
        amlc = max(amlc, c)
        amlt = max(amlt, t)
    return (cmlc, cmlt, amlc, amlt)