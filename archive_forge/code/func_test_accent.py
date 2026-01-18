from bokeh.document import Document
from panel.template.fast.list import FastListTemplate
from panel.theme.fast import FastDarkTheme
def test_accent():
    accent = 'yellow'
    template = FastListTemplate(accent=accent)
    assert template.accent_base_color == accent
    assert template.header_background == accent