import pytest
from packaging.version import Version
import numpy as np
import panel as pn
from panel.models.vega import VegaPlot
from panel.pane import PaneBase, Vega
def altair_example():
    import altair as alt
    data = alt.Data(values=[{'x': 'A', 'y': 5}, {'x': 'B', 'y': 3}, {'x': 'C', 'y': 6}, {'x': 'D', 'y': 7}, {'x': 'E', 'y': 2}])
    chart = alt.Chart(data).mark_bar().encode(x='x:O', y='y:Q')
    return chart