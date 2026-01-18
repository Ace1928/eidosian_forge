import pytest
from bokeh.plotting import figure
from panel.layout import Row
from panel.links import Link
from panel.pane import Bokeh, HoloViews
from panel.tests.util import hv_available
from panel.widgets import (
def test_bokeh_figure_jslink(document, comm):
    fig = figure()
    fig.line(0, 0)
    pane = Bokeh(fig)
    t1 = FloatInput()
    pane.jslink(t1, **{'x_range.start': 'value'})
    row = Row(pane, t1)
    model = row.get_root(document, comm)
    link_customjs = fig.x_range.js_property_callbacks['change:start'][-1]
    assert link_customjs.args['source'] == fig.x_range
    assert link_customjs.args['target'] == model.children[1]
    assert link_customjs.code == "\n    var value = source['start'];\n    value = value;\n    value = value;\n    try {\n      var property = target.properties['value'];\n      if (property !== undefined) { property.validate(value); }\n    } catch(err) {\n      console.log('WARNING: Could not set value on target, raised error: ' + err);\n      return;\n    }\n    try {\n      target['value'] = value;\n    } catch(err) {\n      console.log(err)\n    }\n    "