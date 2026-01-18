from ncclient import operations
from ncclient import transport
import socket
import logging
import functools
from ncclient.xml_ import *
def make_device_handler(device_params):
    """
    Create a device handler object that provides device specific parameters and
    functions, which are called in various places throughout our code.

    If no device_params are defined or the "name" in the parameter dict is not
    known then a default handler will be returned.

    """
    if device_params is None:
        device_params = {}
    handler = device_params.get('handler', None)
    if handler:
        return handler(device_params)
    device_name = device_params.get('name', 'default')
    class_name = '%sDeviceHandler' % device_name.capitalize()
    devices_module_name = 'ncclient.devices.%s' % device_name
    dev_module_obj = __import__(devices_module_name)
    handler_module_obj = getattr(getattr(dev_module_obj, 'devices'), device_name)
    class_obj = getattr(handler_module_obj, class_name)
    handler_obj = class_obj(device_params)
    return handler_obj