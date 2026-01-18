import pytest
from panel import FloatPanel, Row, Spacer
from panel.tests.util import serve_component, wait_until
def test_float_panel_resize(page):
    float_panel = FloatPanel(Row(Spacer(styles=dict(background='red'), css_classes=['red'], height=200, sizing_mode='stretch_width'), Spacer(styles=dict(background='green'), css_classes=['green'], height=200, sizing_mode='stretch_width'), Spacer(styles=dict(background='blue'), css_classes=['blue'], height=200, sizing_mode='stretch_width')))
    serve_component(page, float_panel)
    resize_handle = page.locator('.jsPanel-resizeit-se')
    resize_handle.drag_to(resize_handle, target_position={'x': 510, 'y': 300}, force=True)
    wait_until(lambda: int(page.locator('.red').bounding_box()['width']) == 200, page)
    wait_until(lambda: int(page.locator('.green').bounding_box()['width']) == 200, page)
    wait_until(lambda: int(page.locator('.blue').bounding_box()['width']) == 200, page)