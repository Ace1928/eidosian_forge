from holoviews.element import (
from .geo import (_Element, Feature, Tiles, is_geographic,     # noqa (API import)
def filledcontours(self, kdims=None, vdims=None, mdims=None, **kwargs):
    return self(FilledContours, kdims, vdims, mdims, **kwargs)