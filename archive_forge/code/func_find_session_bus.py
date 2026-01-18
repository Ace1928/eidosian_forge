import os
import re
def find_session_bus():
    addr = os.environ['DBUS_SESSION_BUS_ADDRESS']
    return next(get_connectable_addresses(addr))