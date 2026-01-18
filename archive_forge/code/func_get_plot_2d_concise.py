from __future__ import annotations
import json
import os
from collections import namedtuple
from fractions import Fraction
from typing import TYPE_CHECKING, cast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.constants as sc
from pymatgen.analysis.diffraction.core import AbstractDiffractionPatternCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.string import latexify_spacegroup, unicodeify_spacegroup
def get_plot_2d_concise(self, structure: Structure) -> go.Figure:
    """
        Generates the concise 2D diffraction pattern of the input structure of a smaller size and without layout.
        Does not display.

        Args:
            structure (Structure): The input structure.

        Returns:
            Figure
        """
    if self.symprec:
        finder = SpacegroupAnalyzer(structure, symprec=self.symprec)
        structure = finder.get_refined_structure()
    points = self.generate_points(-10, 11)
    tem_dots = self.tem_dots(structure, points)
    xs = []
    ys = []
    hkls = []
    intensities = []
    for dot in tem_dots:
        if dot.hkl != (0, 0, 0):
            xs.append(dot.position[0])
            ys.append(dot.position[1])
            hkls.append(dot.hkl)
            intensities.append(dot.intensity)
    data = [go.Scatter(x=xs, y=ys, text=hkls, mode='markers', hoverinfo='skip', marker={'size': 4, 'cmax': 1, 'cmin': 0, 'color': intensities, 'colorscale': [[0, 'black'], [1, 'white']]}, showlegend=False)]
    layout = dict(xaxis={'range': [-4, 4], 'showgrid': False, 'zeroline': False, 'showline': False, 'ticks': '', 'showticklabels': False}, yaxis={'range': [-4, 4], 'showgrid': False, 'zeroline': False, 'showline': False, 'ticks': '', 'showticklabels': False}, plot_bgcolor='black', margin={'l': 0, 'r': 0, 't': 0, 'b': 0}, width=121, height=121)
    fig = go.Figure(data=data, layout=layout)
    fig.layout.update(showlegend=False)
    return fig