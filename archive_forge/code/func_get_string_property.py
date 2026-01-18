from __future__ import absolute_import
import ctypes
from serial.tools import list_ports_common
def get_string_property(device_type, property):
    """
    Search the given device for the specified string property

    @param device_type Type of Device
    @param property String to search for
    @return Python string containing the value, or None if not found.
    """
    key = cf.CFStringCreateWithCString(kCFAllocatorDefault, property.encode('utf-8'), kCFStringEncodingUTF8)
    CFContainer = iokit.IORegistryEntryCreateCFProperty(device_type, key, kCFAllocatorDefault, 0)
    output = None
    if CFContainer:
        output = cf.CFStringGetCStringPtr(CFContainer, 0)
        if output is not None:
            output = output.decode('utf-8')
        else:
            buffer = ctypes.create_string_buffer(io_name_size)
            success = cf.CFStringGetCString(CFContainer, ctypes.byref(buffer), io_name_size, kCFStringEncodingUTF8)
            if success:
                output = buffer.value.decode('utf-8')
        cf.CFRelease(CFContainer)
    return output