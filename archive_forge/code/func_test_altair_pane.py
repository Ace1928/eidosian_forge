import pytest
from packaging.version import Version
import numpy as np
import panel as pn
from panel.models.vega import VegaPlot
from panel.pane import PaneBase, Vega
@altair_available
def test_altair_pane(document, comm):
    pane = Vega(altair_example())
    model = pane.get_root(document, comm=comm)
    assert isinstance(model, VegaPlot)
    expected = dict(vega_example, data={})
    if altair_version >= Version('5.0.0rc1'):
        expected['mark'] = {'type': 'bar'}
        expected['config'] = vega5_config
    elif altair_version >= Version('4.0.0'):
        expected['config'] = vega4_config
    assert dict(model.data, **blank_schema) == dict(expected, **blank_schema)
    cds_data = model.data_sources['data'].data
    assert np.array_equal(cds_data['x'], np.array(['A', 'B', 'C', 'D', 'E']))
    assert np.array_equal(cds_data['y'], np.array([5, 3, 6, 7, 2]))
    chart = altair_example()
    chart.mark = 'point'
    chart.data.values[0]['x'] = 'C'
    pane.object = chart
    point_example = dict(vega_example, data={}, mark='point')
    if altair_version >= Version('5.0.0rc1'):
        point_example['mark'] = {'type': 'point'}
        point_example['config'] = vega5_config
    elif altair_version >= Version('4.0.0'):
        point_example['config'] = vega4_config
    assert dict(model.data, **blank_schema) == dict(point_example, **blank_schema)
    cds_data = model.data_sources['data'].data
    assert np.array_equal(cds_data['x'], np.array(['C', 'B', 'C', 'D', 'E']))
    assert np.array_equal(cds_data['y'], np.array([5, 3, 6, 7, 2]))
    pane._cleanup(model)
    assert pane._models == {}