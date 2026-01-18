import warnings
import numpy as np
from numpy import cos, sin, pi
def klein_bottle(draw=True, show=True, figure8=False, endpoint=True, uv=True, wireframe=False, texture=None, both=False, interval=1000, **kwargs):
    """Show one or two Klein bottles."""
    import ipyvolume.pylab as p3
    u = np.linspace(0, 2 * pi, num=40, endpoint=endpoint)
    v = np.linspace(0, 2 * pi, num=40, endpoint=endpoint)
    u, v = np.meshgrid(u, v)
    if both:
        x1, y1, z1, _u1, _v1 = klein_bottle(endpoint=endpoint, draw=False, show=False, **kwargs)
        x2, y2, z2, _u2, _v2 = klein_bottle(endpoint=endpoint, draw=False, show=False, figure8=True, **kwargs)
        x = [x1, x2]
        y = [y1, y2]
        z = [z1, z2]
    elif figure8:
        a = 2
        s = 5
        x = s * (a + cos(u / 2) * sin(v) - sin(u / 2) * sin(2 * v) / 2) * cos(u)
        y = s * (a + cos(u / 2) * sin(v) - sin(u / 2) * sin(2 * v) / 2) * sin(u)
        z = s * (sin(u / 2) * sin(v) + cos(u / 2) * sin(2 * v) / 2)
    else:
        r = 4 * (1 - cos(u) / 2)
        x = 6 * cos(u) * (1 + sin(u)) + r * cos(u) * cos(v) * (u < pi) + r * cos(v + pi) * (u >= pi)
        y = 16 * sin(u) + r * sin(u) * cos(v) * (u < pi)
        z = r * sin(v)
        x = x / 20
        y = y / 20
        z = z / 20
    if draw:
        if texture:
            uv = True
        if uv:
            mesh = p3.plot_mesh(x, y, z, wrapx=not endpoint, wrapy=not endpoint, u=u / (2 * np.pi), v=v / (2 * np.pi), wireframe=wireframe, texture=texture, **kwargs)
        else:
            mesh = p3.plot_mesh(x, y, z, wrapx=not endpoint, wrapy=not endpoint, wireframe=wireframe, texture=texture, **kwargs)
        if show:
            if both:
                p3.animation_control(mesh, interval=interval)
            p3.squarelim()
            p3.show()
        return mesh
    else:
        return (x, y, z, u, v)