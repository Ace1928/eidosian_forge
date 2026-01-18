import numpy as np
from ..Qt import QtGui
@staticmethod
def cylinder(rows, cols, radius=[1.0, 1.0], length=1.0, offset=False):
    """
        Return a MeshData instance with vertexes and faces computed
        for a cylindrical surface.
        The cylinder may be tapered with different radii at each end (truncated cone)
        """
    verts = np.empty((rows + 1, cols, 3), dtype=float)
    if isinstance(radius, int):
        radius = [radius, radius]
    th = np.linspace(2 * np.pi, 2 * np.pi / cols, cols).reshape(1, cols)
    r = np.linspace(radius[0], radius[1], num=rows + 1, endpoint=True).reshape(rows + 1, 1)
    verts[..., 2] = np.linspace(0, length, num=rows + 1, endpoint=True).reshape(rows + 1, 1)
    if offset:
        th = th + np.pi / cols * np.arange(rows + 1).reshape(rows + 1, 1)
    verts[..., 0] = r * np.cos(th)
    verts[..., 1] = r * np.sin(th)
    verts = verts.reshape((rows + 1) * cols, 3)
    faces = np.empty((rows * cols * 2, 3), dtype=np.uint)
    rowtemplate1 = (np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 0]])) % cols + np.array([[0, 0, cols]])
    rowtemplate2 = (np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 1]])) % cols + np.array([[cols, 0, cols]])
    for row in range(rows):
        start = row * cols * 2
        faces[start:start + cols] = rowtemplate1 + row * cols
        faces[start + cols:start + cols * 2] = rowtemplate2 + row * cols
    return MeshData(vertexes=verts, faces=faces)