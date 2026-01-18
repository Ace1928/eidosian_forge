from __future__ import annotations
import importlib
import math
from functools import wraps
from string import ascii_letters
from typing import TYPE_CHECKING, Literal
import matplotlib.pyplot as plt
import numpy as np
import palettable.colorbrewer.diverging
from matplotlib import cm, colors
from pymatgen.core import Element
def periodic_table_heatmap(elemental_data=None, cbar_label='', cbar_label_size=14, show_plot: bool=False, cmap='YlOrRd', cmap_range=None, blank_color='grey', edge_color='white', value_format=None, value_fontsize=10, symbol_fontsize=14, max_row: int=9, readable_fontcolor=False, pymatviz: bool=True, **kwargs):
    """A static method that generates a heat map overlaid on a periodic table.

    Args:
        elemental_data (dict): A dictionary with the element as a key and a
            value assigned to it, e.g. surface energy and frequency, etc.
            Elements missing in the elemental_data will be grey by default
            in the final table elemental_data={"Fe": 4.2, "O": 5.0}.
        cbar_label (str): Label of the color bar. Default is "".
        cbar_label_size (float): Font size for the color bar label. Default is 14.
        cmap_range (tuple): Minimum and maximum value of the color map scale.
            If None, the color map will automatically scale to the range of the
            data.
        show_plot (bool): Whether to show the heatmap. Default is False.
        value_format (str): Formatting string to show values. If None, no value
            is shown. Example: "%.4f" shows float to four decimals.
        value_fontsize (float): Font size for values. Default is 10.
        symbol_fontsize (float): Font size for element symbols. Default is 14.
        cmap (str): Color scheme of the heatmap. Default is 'YlOrRd'.
            Refer to the matplotlib documentation for other options.
        blank_color (str): Color assigned for the missing elements in
            elemental_data. Default is "grey".
        edge_color (str): Color assigned for the edge of elements in the
            periodic table. Default is "white".
        max_row (int): Maximum number of rows of the periodic table to be
            shown. Default is 9, which means the periodic table heat map covers
            the standard 7 rows of the periodic table + 2 rows for the lanthanides
            and actinides. Use a value of max_row = 7 to exclude the lanthanides and
            actinides.
        readable_fontcolor (bool): Whether to use readable font color depending
            on background color. Default is False.
        pymatviz (bool): Whether to use pymatviz to generate the heatmap. Defaults to True.
            See https://github.com/janosh/pymatviz.
        kwargs: Passed to pymatviz.ptable_heatmap_plotly

    Returns:
        plt.Axes: matplotlib Axes object
    """
    if pymatviz:
        try:
            from pymatviz import ptable_heatmap_plotly
            if elemental_data:
                kwargs.setdefault('values', elemental_data)
                print('elemental_data is deprecated, use values={"Fe": 4.2, "O": 5.0} instead')
            if cbar_label:
                kwargs.setdefault('color_bar', {}).setdefault('title', cbar_label)
                print('cbar_label is deprecated, use color_bar={"title": cbar_label} instead')
            if cbar_label_size != 14:
                kwargs.setdefault('color_bar', {}).setdefault('titlefont', {}).setdefault('size', cbar_label_size)
                print('cbar_label_size is deprecated, use color_bar={"titlefont": {"size": cbar_label_size}} instead')
            if cmap:
                kwargs.setdefault('colorscale', cmap)
                print('cmap is deprecated, use colorscale=cmap instead')
            if cmap_range:
                kwargs.setdefault('cscale_range', cmap_range)
                print('cmap_range is deprecated, use cscale_range instead')
            if value_format:
                kwargs.setdefault('fmt', value_format)
                print('value_format is deprecated, use fmt instead')
            if blank_color != 'grey':
                print('blank_color is deprecated')
            if edge_color != 'white':
                print('edge_color is deprecated')
            if symbol_fontsize != 14:
                print('symbol_fontsize is deprecated, use font_size instead')
                kwargs.setdefault('font_size', symbol_fontsize)
            if value_fontsize != 10:
                print('value_fontsize is deprecated, use font_size instead')
                kwargs.setdefault('font_size', value_fontsize)
            if max_row != 9:
                print('max_row is deprecated, use max_row instead')
            if readable_fontcolor:
                print("readable_fontcolor is deprecated, use font_colors instead, e.g. ('black', 'white')")
            return ptable_heatmap_plotly(**kwargs)
        except ImportError:
            print("You're using a deprecated version of periodic_table_heatmap(). Consider `pip install pymatviz` which offers an interactive plotly periodic table heatmap. You can keep calling this same function from pymatgen. Some of the arguments have changed which you'll be warned about. To disable this warning, pass pymatviz=False.")
    if cmap_range is not None:
        max_val = cmap_range[1]
        min_val = cmap_range[0]
    else:
        max_val = max(elemental_data.values())
        min_val = min(elemental_data.values())
    max_row = min(max_row, 9)
    if max_row <= 0:
        raise ValueError("The input argument 'max_row' must be positive!")
    value_table = np.empty((max_row, 18)) * np.nan
    blank_value = min_val - 0.01
    for el in Element:
        value = elemental_data.get(el.symbol, blank_value)
        if 57 <= el.Z <= 71:
            plot_row = 8
            plot_group = (el.Z - 54) % 32
        elif 89 <= el.Z <= 103:
            plot_row = 9
            plot_group = (el.Z - 54) % 32
        else:
            plot_row = el.row
            plot_group = el.group
        if plot_row > max_row:
            continue
        value_table[plot_row - 1, plot_group - 1] = value
    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(12, 8)
    data_mask = np.ma.masked_invalid(value_table.tolist())
    heatmap = ax.pcolor(data_mask, cmap=cmap, edgecolors=edge_color, linewidths=1, vmin=min_val - 0.001, vmax=max_val + 0.001)
    cbar = fig.colorbar(heatmap)
    cbar.cmap.set_under(blank_color)
    cbar.set_label(cbar_label, rotation=270, labelpad=25, size=cbar_label_size)
    cbar.ax.tick_params(labelsize=cbar_label_size)
    ax.axis('off')
    ax.invert_yaxis()
    norm = colors.Normalize(vmin=min_val, vmax=max_val)
    scalar_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    for ii, row in enumerate(value_table):
        for jj, el in enumerate(row):
            if not np.isnan(el):
                symbol = Element.from_row_and_group(ii + 1, jj + 1).symbol
                rgba = scalar_cmap.to_rgba(el)
                fontcolor = _decide_fontcolor(rgba) if readable_fontcolor else 'black'
                plt.text(jj + 0.5, ii + 0.25, symbol, horizontalalignment='center', verticalalignment='center', fontsize=symbol_fontsize, color=fontcolor)
                if el != blank_value and value_format is not None:
                    plt.text(jj + 0.5, ii + 0.5, value_format % el, horizontalalignment='center', verticalalignment='center', fontsize=value_fontsize, color=fontcolor)
    plt.tight_layout()
    if show_plot:
        plt.show()
    return ax