import pytest
from bokeh.plotting import figure
from panel.layout import Row
from panel.links import Link
from panel.pane import Bokeh, HoloViews
from panel.tests.util import hv_available
from panel.widgets import (
@hv_available
def test_bkwidget_hvplot_links(document, comm):
    from bokeh.models import Slider
    bokeh_widget = Slider(value=5, start=1, end=10, step=0.1)
    points1 = hv.Points([1, 2, 3])
    Link(bokeh_widget, points1, properties={'value': 'glyph.size'})
    row = Row(HoloViews(points1, backend='bokeh'), bokeh_widget)
    model = row.get_root(document, comm=comm)
    hv_views = row.select(HoloViews)
    assert len(hv_views) == 1
    slider = bokeh_widget
    scatter = hv_views[0]._plots[model.ref['id']][0].handles['glyph']
    link_customjs = slider.js_property_callbacks['change:value'][-1]
    assert link_customjs.args['source'] is slider
    assert link_customjs.args['target'] is scatter
    code = "\n    var value = source['value'];\n    value = value;\n    value = value;\n    try {\n      var property = target.properties['size'];\n      if (property !== undefined) { property.validate(value); }\n    } catch(err) {\n      console.log('WARNING: Could not set size on target, raised error: ' + err);\n      return;\n    }\n    try {\n      target['size'] = value;\n    } catch(err) {\n      console.log(err)\n    }\n    "
    assert link_customjs.code == code