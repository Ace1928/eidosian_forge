from ncclient import operations
from ncclient import transport
import socket
import logging
import functools
from ncclient.xml_ import *
def connect_ioproc(*args, **kwds):
    device_params = _extract_device_params(kwds)
    manager_params = _extract_manager_params(kwds)
    if device_params:
        import_string = 'ncclient.transport.third_party.'
        import_string += device_params['name'] + '.ioproc'
        third_party_import = __import__(import_string, fromlist=['IOProc'])
    device_handler = make_device_handler(device_params)
    session = third_party_import.IOProc(device_handler)
    session.connect()
    return Manager(session, device_handler, **manager_params)