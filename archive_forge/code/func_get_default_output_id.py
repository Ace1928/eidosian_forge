from midi devices.  It can also list midi devices on the system.
import math
import atexit
import pygame
import pygame.locals
import pygame.pypm as _pypm
def get_default_output_id():
    """gets default output device number
    pygame.midi.get_default_output_id(): return default_id


    Return the default device ID or -1 if there are no devices.
    The result can be passed to the Input()/Output() class.

    On the PC, the user can specify a default device by
    setting an environment variable. For example, to use device #1.

        set PM_RECOMMENDED_OUTPUT_DEVICE=1

    The user should first determine the available device ID by using
    the supplied application "testin" or "testout".

    In general, the registry is a better place for this kind of info,
    and with USB devices that can come and go, using integers is not
    very reliable for device identification. Under Windows, if
    PM_RECOMMENDED_OUTPUT_DEVICE (or PM_RECOMMENDED_INPUT_DEVICE) is
    *NOT* found in the environment, then the default device is obtained
    by looking for a string in the registry under:
        HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Input_Device
    and HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Output_Device
    for a string. The number of the first device with a substring that
    matches the string exactly is returned. For example, if the string
    in the registry is "USB", and device 1 is named
    "In USB MidiSport 1x1", then that will be the default
    input because it contains the string "USB".

    In addition to the name, get_device_info() returns "interf", which
    is the interface name. (The "interface" is the underlying software
    system or API used by PortMidi to access devices. Examples are
    MMSystem, DirectX (not implemented), ALSA, OSS (not implemented), etc.)
    At present, the only Win32 interface is "MMSystem", the only Linux
    interface is "ALSA", and the only Max OS X interface is "CoreMIDI".
    To specify both the interface and the device name in the registry,
    separate the two with a comma and a space, e.g.:
        MMSystem, In USB MidiSport 1x1
    In this case, the string before the comma must be a substring of
    the "interf" string, and the string after the space must be a
    substring of the "name" name string in order to match the device.

    Note: in the current release, the default is simply the first device
    (the input or output device with the lowest PmDeviceID).
    """
    _check_init()
    return _pypm.GetDefaultOutputDeviceID()