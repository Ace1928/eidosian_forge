import pytest
from testpath import modified_env
from jeepney import bus
def test_get_bus():
    with modified_env({'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/1000/bus', 'DBUS_SYSTEM_BUS_ADDRESS': 'unix:path=/var/run/dbus/system_bus_socket'}):
        assert bus.get_bus('SESSION') == '/run/user/1000/bus'
        assert bus.get_bus('SYSTEM') == '/var/run/dbus/system_bus_socket'
    assert bus.get_bus('unix:path=/run/user/1002/bus') == '/run/user/1002/bus'