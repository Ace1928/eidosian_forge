import pytest
from jeepney import new_method_call, MessageType, DBusAddress
from jeepney.bus_messages import message_bus, MatchRule
from jeepney.io.blocking import open_dbus_connection, Proxy
from .utils import have_session_bus
@pytest.fixture
def session_conn():
    with open_dbus_connection(bus='SESSION') as conn:
        yield conn