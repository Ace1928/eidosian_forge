from holoviews.element import (
from .geo import (_Element, Feature, Tiles, is_geographic,     # noqa (API import)
def linecontours(self, kdims=None, vdims=None, mdims=None, **kwargs):
    return self(LineContours, kdims, vdims, mdims, **kwargs)