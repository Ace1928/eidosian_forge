from ipywidgets import register
from .Geometry_autogen import Geometry as AutogenGeometry
from .._base.Three import ThreeWidget
@classmethod
def from_geometry(cls, geometry, store_ref=False):
    """Creates a PlainGeometry of another geometry.

        store_ref determines if the reference is stored after initalization.
        If it is, it will be used for future embedding.

        NOTE:
        The PlainGeometry will copy the arrays from the source geometry.
        To avoid this, use PlainBufferGeometry.
        """
    return cls(_ref_geometry=geometry, _store_ref=store_ref)