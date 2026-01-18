import os
import re
def get_bus(addr):
    if addr == 'SESSION':
        return find_session_bus()
    elif addr == 'SYSTEM':
        return find_system_bus()
    else:
        return next(get_connectable_addresses(addr))