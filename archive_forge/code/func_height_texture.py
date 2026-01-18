from ipywidgets import Widget, widget_serialization
from traitlets import Unicode, CInt, Instance, List, CFloat, Bool, observe, validate
import numpy as np
from ._package import npm_pkg_name
from ._version import EXTENSION_SPEC_VERSION
from .core.BufferAttribute import BufferAttribute
from .core.Geometry import Geometry
from .core.BufferGeometry import BufferGeometry
from .geometries.BoxGeometry_autogen import BoxGeometry
from .geometries.SphereGeometry_autogen import SphereGeometry
from .lights.AmbientLight_autogen import AmbientLight
from .lights.DirectionalLight_autogen import DirectionalLight
from .materials.Material_autogen import Material
from .materials.MeshLambertMaterial_autogen import MeshLambertMaterial
from .materials.SpriteMaterial_autogen import SpriteMaterial
from .objects.Group_autogen import Group
from .objects.Line_autogen import Line
from .objects.Mesh_autogen import Mesh
from .objects.Sprite_autogen import Sprite
from .textures.Texture_autogen import Texture
from .textures.DataTexture import DataTexture
from .textures.TextTexture_autogen import TextTexture
def height_texture(z, colormap='viridis'):
    """Create a texture corresponding to the heights in z and the given colormap."""
    from matplotlib import cm
    from skimage import img_as_ubyte
    colormap = cm.get_cmap(colormap)
    im = z.copy()
    im -= np.nanmin(im)
    im /= np.nanmax(im)
    im = np.nan_to_num(im)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Possible precision loss when converting from', category=UserWarning, module='skimage.util.dtype')
        rgba_im = img_as_ubyte(colormap(im))
    return DataTexture(data=rgba_im, format='RGBAFormat')