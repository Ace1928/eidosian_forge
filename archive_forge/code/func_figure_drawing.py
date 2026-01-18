from pyomo.common.dependencies import numpy as np, pandas as pd, matplotlib as plt
from pyomo.core.expr.numvalue import value
from itertools import product
import logging
from pyomo.opt import SolverStatus, TerminationCondition
def figure_drawing(self, fixed_design_dimensions, sensitivity_dimension, title_text, xlabel_text, ylabel_text, font_axes=16, font_tick=14, log_scale=True):
    """
        Extract results needed for drawing figures from the overall result dataframe.
        Draw 1D sensitivity curve or 2D heatmap.
        It can be applied to results of any dimensions, but requires design variable values in other dimensions be fixed.

        Parameters
        ----------
        fixed_design_dimensions: a dictionary, keys are the design variable names to be fixed, values are the value of it to be fixed.
        sensitivity_dimension: a list of design variable names to draw figures.
            If only one name is given, a 1D sensitivity curve is drawn
            if two names are given, a 2D heatmap is drawn.
        title_text: name of the figure, a string
        xlabel_text: x label title, a string.
            In a 1D sensitivity curve, it is the design variable by which the curve is drawn.
            In a 2D heatmap, it should be the second design variable in the design_ranges
        ylabel_text: y label title, a string.
            A 1D sensitivity curve does not need it. In a 2D heatmap, it should be the first design variable in the dv_ranges
        font_axes: axes label font size
        font_tick: tick label font size
        log_scale: if True, the result matrix will be scaled by log10

        Returns
        --------
        None
        """
    self.fixed_design_names = list(fixed_design_dimensions.keys())
    self.fixed_design_values = list(fixed_design_dimensions.values())
    self.sensitivity_dimension = sensitivity_dimension
    if len(self.fixed_design_names) + len(self.sensitivity_dimension) != len(self.design_names):
        raise ValueError('Error: All dimensions except for those the figures are drawn by should be fixed.')
    if len(self.sensitivity_dimension) not in [1, 2]:
        raise ValueError('Error: Either 1D or 2D figures can be drawn.')
    if len(self.fixed_design_names) != 0:
        filter = ''
        for i in range(len(self.fixed_design_names)):
            filter += '(self.store_all_results_dataframe['
            filter += str(self.fixed_design_names[i])
            filter += ']=='
            filter += str(self.fixed_design_values[i])
            filter += ')'
            if i != len(self.fixed_design_names) - 1:
                filter += '&'
        figure_result_data = self.store_all_results_dataframe.loc[eval(filter)]
    else:
        figure_result_data = self.store_all_results_dataframe
    self.figure_result_data = figure_result_data
    if len(sensitivity_dimension) == 1:
        self._curve1D(title_text, xlabel_text, font_axes=16, font_tick=14, log_scale=True)
    elif len(sensitivity_dimension) == 2:
        self._heatmap(title_text, xlabel_text, ylabel_text, font_axes=16, font_tick=14, log_scale=True)