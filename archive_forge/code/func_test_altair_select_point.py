import pytest
from playwright.sync_api import expect
from panel.pane import Vega
from panel.tests.util import serve_component, wait_until
@altair_available
def test_altair_select_point(page, dataframe):
    multi = alt.selection_point(name='multi')
    chart = alt.Chart(dataframe).mark_point(size=12000).encode(x='int', y='float', color=alt.condition(multi, alt.value('black'), alt.value('lightgray'))).add_params(multi)
    pane = Vega(chart)
    serve_component(page, pane)
    vega_plot = page.locator('.vega-embed')
    expect(vega_plot).to_have_count(1)
    bbox = vega_plot.bounding_box()
    page.mouse.click(bbox['x'] + 200, bbox['y'] + 150)
    wait_until(lambda: pane.selection.multi == [2], page)
    page.keyboard.down('Shift')
    page.mouse.click(bbox['x'] + 300, bbox['y'] + 100)
    wait_until(lambda: pane.selection.multi == [2, 3], page)