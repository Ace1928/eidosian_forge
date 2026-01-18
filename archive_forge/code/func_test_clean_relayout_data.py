import datetime as dt
import pytest
import numpy as np
import pandas as pd
from panel.models.plotly import PlotlyPlot
from panel.pane import PaneBase, Plotly
@plotly_available
def test_clean_relayout_data():
    relayout_data = {'mapbox.center': {'lon': -73.59183434290809, 'lat': 45.52341668343991}, 'mapbox.zoom': 10, 'mapbox.bearing': 0, 'mapbox.pitch': 0, 'mapbox._derived': {'coordinates': [[-73.92279747767401, 45.597934047192865], [-73.26087120814279, 45.597934047192865], [-73.26087120814279, 45.44880048640681], [-73.92279747767401, 45.44880048640681]]}}
    result = Plotly._clean_relayout_data(relayout_data)
    assert result == {'mapbox.center': {'lon': -73.59183434290809, 'lat': 45.52341668343991}, 'mapbox.zoom': 10, 'mapbox.bearing': 0, 'mapbox.pitch': 0}