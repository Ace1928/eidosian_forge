from __future__ import annotations
import itertools
import math
import os
import subprocess
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from pymatgen.core import PeriodicSite, Species, Structure
from pymatgen.util.coord import in_coord_list
def make_movie(structures, output_filename='movie.mp4', zoom=1.0, fps=20, bitrate='10000k', quality=1, **kwargs):
    """
    Generate a movie from a sequence of structures using vtk and ffmpeg.

    Args:
        structures ([Structure]): sequence of structures
        output_filename (str): filename for structure output. defaults to
            movie.mp4
        zoom (float): A zoom to be applied to the visualizer. Defaults to 1.0.
        fps (int): Frames per second for the movie. Defaults to 20.
        bitrate (str): Video bitrate. Defaults to "10000k" (fairly high
            quality).
        quality (int): A quality scale. Defaults to 1.
        kwargs: Any kwargs supported by StructureVis to modify the images
            generated.
    """
    vis = StructureVis(**kwargs)
    vis.show_help = False
    vis.redraw()
    vis.zoom(zoom)
    sig_fig = int(math.floor(math.log10(len(structures))) + 1)
    filename = f'image{{0:0{sig_fig}d}}.png'
    for idx, site in enumerate(structures):
        vis.set_structure(site)
        vis.write_image(filename.format(idx), 3)
    filename = f'image%0{sig_fig}d.png'
    args = ['ffmpeg', '-y', '-i', filename, '-q:v', str(quality), '-r', str(fps), '-b:v', str(bitrate), output_filename]
    with subprocess.Popen(args) as p:
        p.communicate()