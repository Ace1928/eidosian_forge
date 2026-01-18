import glob
import json
import os
from io import StringIO
import pytest
from bokeh.models import CustomJS
from panel import Row
from panel.config import config
from panel.io.embed import embed_state
from panel.pane import Str
from panel.param import Param
from panel.widgets import (
def test_embed_slider_str_jslink(document, comm):
    slider = FloatSlider(start=0, end=10)
    string = Str()
    slider.link(string, value='object')
    panel = Row(slider, string)
    with config.set(embed=True):
        model = panel.get_root(document, comm)
    embed_state(panel, model, document)
    assert len(document.roots) == 1
    assert model is document.roots[0]
    ref = model.ref['id']
    cbs = list(model.select({'type': CustomJS}))
    assert len(cbs) == 2
    cb1, cb2 = cbs
    cb1, cb2 = (cb1, cb2) if slider._models[ref][0] is cb1.args['source'] else (cb2, cb1)
    assert cb1.code == '\n    var value = source[\'value\'];\n    value = value;\n    value = JSON.stringify(value).replace(/,/g, ", ").replace(/:/g, ": ");\n    try {\n      var property = target.properties[\'text\'];\n      if (property !== undefined) { property.validate(value); }\n    } catch(err) {\n      console.log(\'WARNING: Could not set text on target, raised error: \' + err);\n      return;\n    }\n    try {\n      target[\'text\'] = value;\n    } catch(err) {\n      console.log(err)\n    }\n    '
    assert cb2.code == "\n    var value = source['text'];\n    value = value;\n    value = value;\n    try {\n      var property = target.properties['value'];\n      if (property !== undefined) { property.validate(value); }\n    } catch(err) {\n      console.log('WARNING: Could not set value on target, raised error: ' + err);\n      return;\n    }\n    try {\n      target['value'] = value;\n    } catch(err) {\n      console.log(err)\n    }\n    "