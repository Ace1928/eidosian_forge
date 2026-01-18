import json
import param
import pytest
from bokeh.document import Document
from bokeh.io.doc import patch_curdoc
from panel.layout import GridSpec, Row
from panel.pane import HoloViews, Markdown
from panel.template import (
from panel.template.base import BasicTemplate
from panel.widgets import FloatSlider
from .util import hv_available
@pytest.mark.parametrize('template', list_templates)
def test_basic_template(template, document, comm):
    tmplt = template(title='BasicTemplate', header_background='blue', header_color='red')
    tmplt._update_vars()
    tvars = tmplt._render_variables
    assert tvars['app_title'] == 'BasicTemplate'
    assert tvars['header_background'] == 'blue'
    assert tvars['header_color'] == 'red'
    assert tvars['nav'] == False
    assert tvars['busy'] == True
    assert tvars['header'] == False
    titems = tmplt._render_items
    assert titems['busy_indicator'] == (tmplt.busy_indicator, [])
    markdown = Markdown('# Some title')
    tmplt.main.append(markdown)
    assert titems[f'main-{id(markdown)}'] == (markdown, ['main'])
    slider = FloatSlider()
    tmplt.sidebar.append(slider)
    assert titems[f'nav-{id(slider)}'] == (slider, ['nav'])
    assert tvars['nav'] == True
    tmplt.sidebar[:] = []
    assert tvars['nav'] == False
    assert f'nav-{id(slider)}' not in titems
    subtitle = Markdown('## Some subtitle')
    tmplt.header.append(subtitle)
    assert titems[f'header-{id(subtitle)}'] == (subtitle, ['header'])
    assert tvars['header'] == True
    tmplt.header[:] = []
    assert f'header-{id(subtitle)}' not in titems
    assert tvars['header'] == False