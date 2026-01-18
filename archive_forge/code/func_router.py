import pytest
from jeepney import new_method_call, MessageType, DBusAddress
from jeepney.bus_messages import message_bus, MatchRule
from jeepney.io.threading import open_dbus_router, Proxy
from .utils import have_session_bus
@pytest.fixture
def router():
    with open_dbus_router(bus='SESSION') as conn:
        yield conn