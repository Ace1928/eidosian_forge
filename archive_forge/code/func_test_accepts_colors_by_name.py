from bokeh.document import Document
from panel.template.fast.list import FastListTemplate
from panel.theme.fast import FastDarkTheme
def test_accepts_colors_by_name():
    template = FastListTemplate(accent_base_color='red', header_background='green', header_color='white', header_accent_base_color='blue')
    template._update_vars()