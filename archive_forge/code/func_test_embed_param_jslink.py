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
def test_embed_param_jslink(document, comm):
    select = Select(options=['A', 'B', 'C'])
    params = Param(select, parameters=['disabled']).layout
    panel = Row(select, params)
    with config.set(embed=True):
        model = panel.get_root(document, comm)
    embed_state(panel, model, document)
    assert len(document.roots) == 1
    ref = model.ref['id']
    cbs = list(model.select({'type': CustomJS}))
    assert len(cbs) == 2
    cb1, cb2 = cbs
    cb1, cb2 = (cb1, cb2) if select._models[ref][0] is cb1.args['target'] else (cb2, cb1)
    assert cb1.code == "\n    var value = source['active'];\n    value = value;\n    value = value;\n    try {\n      var property = target.properties['disabled'];\n      if (property !== undefined) { property.validate(value); }\n    } catch(err) {\n      console.log('WARNING: Could not set disabled on target, raised error: ' + err);\n      return;\n    }\n    try {\n      target['disabled'] = value;\n    } catch(err) {\n      console.log(err)\n    }\n    "
    assert cb2.code == "\n    var value = source['disabled'];\n    value = value;\n    value = value;\n    try {\n      var property = target.properties['active'];\n      if (property !== undefined) { property.validate(value); }\n    } catch(err) {\n      console.log('WARNING: Could not set active on target, raised error: ' + err);\n      return;\n    }\n    try {\n      target['active'] = value;\n    } catch(err) {\n      console.log(err)\n    }\n    "