from panel.layout import Row
from panel.pane import ECharts, Markdown
def test_echart_js_event_with_arg(document, comm):
    echart = ECharts(ECHART, width=500, height=500)
    md = Markdown()
    echart.js_on_event('click', 'console.log(cb_data)', md=md)
    root = Row(echart, md).get_root(document, comm)
    ref = root.ref['id']
    model = echart._models[ref][0]
    assert model.data == ECHART
    assert 'click' in model.js_events
    assert len(model.js_events['click']) == 1
    handler = model.js_events['click'][0]
    assert handler['callback'].code == 'console.log(cb_data)'
    assert handler['callback'].args == {'md': md._models[ref][0]}