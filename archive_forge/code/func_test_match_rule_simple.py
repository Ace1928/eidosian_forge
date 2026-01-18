from jeepney import DBusAddress, new_signal, new_method_call
from jeepney.bus_messages import MatchRule, message_bus
def test_match_rule_simple():
    rule = MatchRule(type='signal', interface='org.freedesktop.portal.Request')
    assert rule.matches(new_signal(portal_req_iface, 'Response'))
    assert not rule.matches(new_method_call(portal_req_iface, 'Boo'))
    assert not rule.matches(new_signal(portal.with_interface('org.freedesktop.portal.FileChooser'), 'Response'))