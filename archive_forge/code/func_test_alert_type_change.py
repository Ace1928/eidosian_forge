import pytest
import panel as pn
from panel.pane import Alert
from panel.pane.alert import ALERT_TYPES
@pytest.mark.parametrize(['alert_type'], [(alert_type,) for alert_type in ALERT_TYPES])
def test_alert_type_change(alert_type, document, comm):
    """Test that an alert can change alert_type"""
    alert = Alert('This is some text')
    model = alert.get_root(document, comm)
    alert.alert_type = alert_type
    assert set(model.css_classes) == {'alert', f'alert-{alert_type}', 'markdown'}