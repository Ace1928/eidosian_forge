import inspect
import numpy as np
from .bdf import BDF
from .radau import Radau
from .rk import RK23, RK45, DOP853
from .lsoda import LSODA
from scipy.optimize import OptimizeResult
from .common import EPS, OdeSolution
from .base import OdeSolver
def prepare_events(events):
    """Standardize event functions and extract is_terminal and direction."""
    if callable(events):
        events = (events,)
    if events is not None:
        is_terminal = np.empty(len(events), dtype=bool)
        direction = np.empty(len(events))
        for i, event in enumerate(events):
            try:
                is_terminal[i] = event.terminal
            except AttributeError:
                is_terminal[i] = False
            try:
                direction[i] = event.direction
            except AttributeError:
                direction[i] = 0
    else:
        is_terminal = None
        direction = None
    return (events, is_terminal, direction)