import pytest
from playwright.sync_api import expect
from panel.pane import Vega
from panel.tests.util import serve_component, wait_until
@altair_available
def test_altair_select_agg(page, dataframe):
    multi_bar_ref = alt.selection_point(fields=['str'], name='multi_bar_ref')
    chart = alt.Chart(dataframe).mark_bar().transform_aggregate(x='mean(float)', groupby=['str']).encode(x=alt.X('x:Q'), y=alt.Y('str:N'), color=alt.condition(multi_bar_ref, alt.value('black'), alt.value('lightgray'))).add_params(multi_bar_ref)
    pane = Vega(chart)
    serve_component(page, pane)
    vega_plot = page.locator('.vega-embed')
    expect(vega_plot).to_have_count(1)
    bbox = vega_plot.bounding_box()
    page.mouse.click(bbox['x'] + 50, bbox['y'] + 50)
    wait_until(lambda: pane.selection.multi_bar_ref == [{'str': 'C'}], page)
    page.keyboard.down('Shift')
    page.mouse.click(bbox['x'] + 50, bbox['y'] + 10)
    wait_until(lambda: pane.selection.multi_bar_ref == [{'str': 'C'}, {'str': 'A'}], page)