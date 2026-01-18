import numpy as np
import PIL.Image
import pythreejs
import scipy.interpolate
import ipyvolume as ipv
from ipyvolume.datasets import UrlCached
def plot_milkyway(R_sun=8, size=100):
    mw_image = PIL.Image.open(milkyway_image.fetch())
    rescale = 40
    xmw = np.linspace(0, 1, 10)
    ymw = np.linspace(0, 1, 10)
    xmw, ymw = np.meshgrid(xmw, ymw)
    zmw = xmw * 0 + 0.01
    mw = mesh = ipv.plot_mesh((xmw - 0.5) * rescale, (ymw - 0.5) * rescale + R_sun, zmw, u=xmw, v=ymw, texture=mw_image, wireframe=False)
    mw.material.blending = pythreejs.BlendingMode.CustomBlending
    mw.material.blendSrc = pythreejs.BlendFactors.SrcColorFactor
    mw.material.blendDst = pythreejs.BlendFactors.OneFactor
    mw.material.blendEquation = 'AddEquation'
    mw.material.transparent = True
    mw.material.depthWrite = False
    mw.material.alphaTest = 0.1
    ipv.xyzlim(size)
    return mesh