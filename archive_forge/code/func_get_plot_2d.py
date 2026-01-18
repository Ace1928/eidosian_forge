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
def get_plot_2d(self, structure: Structure) -> go.Figure:
    """
        Generates the 2D diffraction pattern of the input structure.

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
        xs.append(dot.position[0])
        ys.append(dot.position[1])
        hkls.append(str(dot.hkl))
        intensities.append(dot.intensity)
    hkls = list(map(unicodeify_spacegroup, list(map(latexify_spacegroup, hkls))))
    data = [go.Scatter(x=xs, y=ys, text=hkls, hoverinfo='text', mode='markers', marker={'size': 8, 'cmax': 1, 'cmin': 0, 'color': intensities, 'colorscale': [[0, 'black'], [1, 'white']]}, showlegend=False), go.Scatter(x=[0], y=[0], text='(0, 0, 0): Direct beam', hoverinfo='text', mode='markers', marker={'size': 14, 'cmax': 1, 'cmin': 0, 'color': 'white'}, showlegend=False)]
    layout = dict(title='2D Diffraction Pattern<br>Beam Direction: ' + ''.join(map(str, self.beam_direction)), font={'size': 14, 'color': '#7f7f7f'}, hovermode='closest', xaxis={'range': [-4, 4], 'showgrid': False, 'zeroline': False, 'showline': False, 'ticks': '', 'showticklabels': False}, yaxis={'range': [-4, 4], 'showgrid': False, 'zeroline': False, 'showline': False, 'ticks': '', 'showticklabels': False}, width=550, height=550, paper_bgcolor='rgba(100,110,110,0.5)', plot_bgcolor='black')
    return go.Figure(data=data, layout=layout)