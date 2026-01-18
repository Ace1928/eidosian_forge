import math
import ctypes
import pygame as pg
def drawcube_old():
    """
    Draw the cube using the old open GL methods pre 3.2 core context.
    """
    allpoints = list(zip(CUBE_POINTS, CUBE_COLORS))
    GL.glBegin(GL.GL_QUADS)
    for face in CUBE_QUAD_VERTS:
        for vert in face:
            pos, color = allpoints[vert]
            GL.glColor3fv(color)
            GL.glVertex3fv(pos)
    GL.glEnd()
    GL.glColor3f(1.0, 1.0, 1.0)
    GL.glBegin(GL.GL_LINES)
    for line in CUBE_EDGES:
        for vert in line:
            pos, color = allpoints[vert]
            GL.glVertex3fv(pos)
    GL.glEnd()