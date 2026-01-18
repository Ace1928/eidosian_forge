import numpy as np
from ..core import Format
def get_bsdf_serializer(options):
    from . import _bsdf as bsdf

    class NDArrayExtension(bsdf.Extension):
        """Copy of BSDF's NDArrayExtension but deal with lazy blobs."""
        name = 'ndarray'
        cls = np.ndarray

        def encode(self, s, v):
            return dict(shape=v.shape, dtype=str(v.dtype), data=v.tobytes())

        def decode(self, s, v):
            return v

    class ImageExtension(bsdf.Extension):
        """We implement two extensions that trigger on the Image classes."""

        def encode(self, s, v):
            return dict(array=v.array, meta=v.meta)

        def decode(self, s, v):
            return Image(v['array'], v['meta'])

    class Image2DExtension(ImageExtension):
        name = 'image2d'
        cls = Image2D

    class Image3DExtension(ImageExtension):
        name = 'image3d'
        cls = Image3D
    exts = [NDArrayExtension, Image2DExtension, Image3DExtension]
    serializer = bsdf.BsdfSerializer(exts, **options)
    return (bsdf, serializer)