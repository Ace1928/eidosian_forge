from panel.layout import Row
from panel.pane import ECharts, Markdown
def test_echart_js_event(document, comm):
    echart = ECharts(ECHART, width=500, height=500)
    echart.js_on_event('click', 'console.log(cb_data)')
    model = echart.get_root(document, comm)
    assert model.data == ECHART
    assert 'click' in model.js_events
    assert len(model.js_events['click']) == 1
    assert model.js_events['click'][0]['callback'].code == 'console.log(cb_data)'