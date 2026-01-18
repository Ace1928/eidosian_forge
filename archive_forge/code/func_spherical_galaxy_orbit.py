import numpy as np
import PIL.Image
import pythreejs
import scipy.interpolate
import ipyvolume as ipv
from ipyvolume.datasets import UrlCached
def spherical_galaxy_orbit(orbit_x, orbit_y, orbit_z, N_stars=100, sigma_r=1, orbit_visible=False, orbit_line_interpolate=5, N_star_orbits=10, color=[255, 220, 200], size_star=1, scatter_kwargs={}):
    """Create a fake galaxy around the points orbit_x/y/z with N_stars around it."""
    if orbit_line_interpolate > 1:
        x = np.linspace(0, 1, len(orbit_x))
        x_smooth = np.linspace(0, 1, len(orbit_x) * orbit_line_interpolate)
        kind = 'quadratic'
        orbit_x_line = scipy.interpolate.interp1d(x, orbit_x, kind)(x_smooth)
        orbit_y_line = scipy.interpolate.interp1d(x, orbit_y, kind)(x_smooth)
        orbit_z_line = scipy.interpolate.interp1d(x, orbit_z, kind)(x_smooth)
    else:
        orbit_x_line = orbit_x
        orbit_y_line = orbit_y
        orbit_z_line = orbit_z
    line = ipv.plot(orbit_x_line, orbit_y_line, orbit_z_line, visible=orbit_visible)
    x = np.repeat(orbit_x, N_stars).reshape((-1, N_stars))
    y = np.repeat(orbit_y, N_stars).reshape((-1, N_stars))
    z = np.repeat(orbit_z, N_stars).reshape((-1, N_stars))
    xr, yr, zr = np.random.normal(0, scale=sigma_r, size=(3, N_stars))
    r = np.sqrt(xr ** 2 + yr ** 2 + zr ** 2)
    for i in range(N_stars):
        a = np.linspace(0, 1, x.shape[0]) * 2 * np.pi * N_star_orbits
        xo = r[i] * np.sin(a)
        yo = r[i] * np.cos(a)
        zo = a * 0
        xo, yo, zo = np.dot(_randomSO3(), [xo, yo, zo])
        x[:, i] += xo
        y[:, i] += yo
        z[:, i] += zo
    sprite = ipv.scatter(x, y, z, texture=radial_sprite((64, 64), color), marker='square_2d', size=size_star, **scatter_kwargs)
    with sprite.material.hold_sync():
        sprite.material.blending = pythreejs.BlendingMode.CustomBlending
        sprite.material.blendSrc = pythreejs.BlendFactors.SrcColorFactor
        sprite.material.blendDst = pythreejs.BlendFactors.OneFactor
        sprite.material.blendEquation = 'AddEquation'
        sprite.material.transparent = True
        sprite.material.depthWrite = False
        sprite.material.alphaTest = 0.1
    return (sprite, line)