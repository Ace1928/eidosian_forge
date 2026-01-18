from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from copy import copy
from types import SimpleNamespace
from fontTools.misc.fixedTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.misc.transform import Transform
from fontTools.pens.transformPen import TransformPen, TransformPointPen
from fontTools.pens.recordingPen import (
@contextmanager
def pushLocation(self, location, reset: bool):
    self.locationStack.append(self.location)
    self.rawLocationStack.append(self.rawLocation)
    if reset:
        self.location = self.originalLocation.copy()
        self.rawLocation = self.defaultLocationNormalized.copy()
    else:
        self.location = self.location.copy()
        self.rawLocation = {}
    self.location.update(location)
    self.rawLocation.update(location)
    try:
        yield None
    finally:
        self.location = self.locationStack.pop()
        self.rawLocation = self.rawLocationStack.pop()