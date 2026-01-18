from __future__ import absolute_import
import re
import ctypes
from ctypes.wintypes import BOOL
from ctypes.wintypes import HWND
from ctypes.wintypes import DWORD
from ctypes.wintypes import WORD
from ctypes.wintypes import LONG
from ctypes.wintypes import ULONG
from ctypes.wintypes import HKEY
from ctypes.wintypes import BYTE
import serial
from serial.win32 import ULONG_PTR
from serial.tools import list_ports_common
def iterate_comports():
    """Return a generator that yields descriptions for serial ports"""
    PortsGUIDs = (GUID * 8)()
    ports_guids_size = DWORD()
    if not SetupDiClassGuidsFromName('Ports', PortsGUIDs, ctypes.sizeof(PortsGUIDs), ctypes.byref(ports_guids_size)):
        raise ctypes.WinError()
    ModemsGUIDs = (GUID * 8)()
    modems_guids_size = DWORD()
    if not SetupDiClassGuidsFromName('Modem', ModemsGUIDs, ctypes.sizeof(ModemsGUIDs), ctypes.byref(modems_guids_size)):
        raise ctypes.WinError()
    GUIDs = PortsGUIDs[:ports_guids_size.value] + ModemsGUIDs[:modems_guids_size.value]
    for index in range(len(GUIDs)):
        bInterfaceNumber = None
        g_hdi = SetupDiGetClassDevs(ctypes.byref(GUIDs[index]), None, NULL, DIGCF_PRESENT)
        devinfo = SP_DEVINFO_DATA()
        devinfo.cbSize = ctypes.sizeof(devinfo)
        index = 0
        while SetupDiEnumDeviceInfo(g_hdi, index, ctypes.byref(devinfo)):
            index += 1
            hkey = SetupDiOpenDevRegKey(g_hdi, ctypes.byref(devinfo), DICS_FLAG_GLOBAL, 0, DIREG_DEV, KEY_READ)
            port_name_buffer = ctypes.create_unicode_buffer(250)
            port_name_length = ULONG(ctypes.sizeof(port_name_buffer))
            RegQueryValueEx(hkey, 'PortName', None, None, ctypes.byref(port_name_buffer), ctypes.byref(port_name_length))
            RegCloseKey(hkey)
            if port_name_buffer.value.startswith('LPT'):
                continue
            szHardwareID = ctypes.create_unicode_buffer(250)
            if not SetupDiGetDeviceInstanceId(g_hdi, ctypes.byref(devinfo), szHardwareID, ctypes.sizeof(szHardwareID) - 1, None):
                if not SetupDiGetDeviceRegistryProperty(g_hdi, ctypes.byref(devinfo), SPDRP_HARDWAREID, None, ctypes.byref(szHardwareID), ctypes.sizeof(szHardwareID) - 1, None):
                    if ctypes.GetLastError() != ERROR_INSUFFICIENT_BUFFER:
                        raise ctypes.WinError()
            szHardwareID_str = szHardwareID.value
            info = list_ports_common.ListPortInfo(port_name_buffer.value, skip_link_detection=True)
            if szHardwareID_str.startswith('USB'):
                m = re.search('VID_([0-9a-f]{4})(&PID_([0-9a-f]{4}))?(&MI_(\\d{2}))?(\\\\(.*))?', szHardwareID_str, re.I)
                if m:
                    info.vid = int(m.group(1), 16)
                    if m.group(3):
                        info.pid = int(m.group(3), 16)
                    if m.group(5):
                        bInterfaceNumber = int(m.group(5))
                    if m.group(7) and re.match('^\\w+$', m.group(7)):
                        info.serial_number = m.group(7)
                    else:
                        info.serial_number = get_parent_serial_number(devinfo.DevInst, info.vid, info.pid)
                loc_path_str = ctypes.create_unicode_buffer(250)
                if SetupDiGetDeviceRegistryProperty(g_hdi, ctypes.byref(devinfo), SPDRP_LOCATION_PATHS, None, ctypes.byref(loc_path_str), ctypes.sizeof(loc_path_str) - 1, None):
                    m = re.finditer('USBROOT\\((\\w+)\\)|#USB\\((\\w+)\\)', loc_path_str.value)
                    location = []
                    for g in m:
                        if g.group(1):
                            location.append('{:d}'.format(int(g.group(1)) + 1))
                        else:
                            if len(location) > 1:
                                location.append('.')
                            else:
                                location.append('-')
                            location.append(g.group(2))
                    if bInterfaceNumber is not None:
                        location.append(':{}.{}'.format('x', bInterfaceNumber))
                    if location:
                        info.location = ''.join(location)
                info.hwid = info.usb_info()
            elif szHardwareID_str.startswith('FTDIBUS'):
                m = re.search('VID_([0-9a-f]{4})\\+PID_([0-9a-f]{4})(\\+(\\w+))?', szHardwareID_str, re.I)
                if m:
                    info.vid = int(m.group(1), 16)
                    info.pid = int(m.group(2), 16)
                    if m.group(4):
                        info.serial_number = m.group(4)
                info.hwid = info.usb_info()
            else:
                info.hwid = szHardwareID_str
            szFriendlyName = ctypes.create_unicode_buffer(250)
            if SetupDiGetDeviceRegistryProperty(g_hdi, ctypes.byref(devinfo), SPDRP_FRIENDLYNAME, None, ctypes.byref(szFriendlyName), ctypes.sizeof(szFriendlyName) - 1, None):
                info.description = szFriendlyName.value
            szManufacturer = ctypes.create_unicode_buffer(250)
            if SetupDiGetDeviceRegistryProperty(g_hdi, ctypes.byref(devinfo), SPDRP_MFG, None, ctypes.byref(szManufacturer), ctypes.sizeof(szManufacturer) - 1, None):
                info.manufacturer = szManufacturer.value
            yield info
        SetupDiDestroyDeviceInfoList(g_hdi)