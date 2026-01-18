from __future__ import absolute_import
import os
import shutil
import json
import contextlib
import numpy as np
import pytest
import ipyvolume
import ipyvolume.pylab as p3
import ipyvolume as ipv
import ipyvolume.examples
import ipyvolume.datasets
import ipyvolume.utils
import ipyvolume.serialize
def test_animation_control():
    ipv.figure()
    n_points = 3
    n_frames = 4
    ar = np.zeros(n_points)
    ar_frames = np.zeros((n_frames, n_points))
    colors_frames = np.zeros((n_frames, n_points, 3))
    scalar = 2
    s = ipv.scatter(x=scalar, y=scalar, z=scalar)
    with pytest.raises(ValueError):
        slider = ipv.animation_control(s, add=False).children[1]
    s = ipv.scatter(x=ar, y=scalar, z=scalar)
    slider = ipv.animation_control(s, add=False).children[1]
    assert slider.max == n_points - 1
    s = ipv.scatter(x=ar_frames, y=scalar, z=scalar)
    slider = ipv.animation_control(s, add=False).children[1]
    assert slider.max == n_frames - 1
    s = ipv.scatter(x=scalar, y=scalar, z=scalar, color=colors_frames)
    slider = ipv.animation_control(s, add=False).children[1]
    assert slider.max == n_frames - 1
    Nx, Ny = (10, 7)
    x = np.arange(Nx)
    y = np.arange(Ny)
    x, y = np.meshgrid(x, y)
    z = x + y
    m = ipv.plot_surface(x, y, z)
    with pytest.raises(ValueError):
        slider = ipv.animation_control(m, add=False).children[1]
    z = [x + y * k for k in range(n_frames)]
    m = ipv.plot_surface(x, y, z)
    slider = ipv.animation_control(m, add=False).children[1]
    assert slider.max == n_frames - 1