import pytest
import panel as pn
from panel.pane import Alert
from panel.pane.alert import ALERT_TYPES
def manualtest_all_view():
    """Test that we can construct and view all Alerts"""
    alerts = []
    for alert_type in ALERT_TYPES:
        text = f'            This is a **{alert_type}** alert with [an example link](https://panel.holoviz.org/).\n            Give it a click if you like.'
        alert = Alert(text, alert_type=alert_type)
        alert_app = pn.Column(alert, pn.Param(alert, parameters=['object', 'alert_type'], widgets={'object': pn.widgets.TextAreaInput}))
        alerts.append(alert_app)
        assert 'alert' in alert.css_classes
        assert f'alert-{alert_type}' in alert.css_classes
    return pn.Column(*alerts, margin=50)