import queue
import rtmidi_python as rtmidi
from ..ports import BaseInput, BaseOutput
def get_devices(api=None, **kwargs):
    devices = {}
    input_names = rtmidi.MidiIn().ports
    output_names = rtmidi.MidiOut().ports
    for name in input_names + output_names:
        if name not in devices:
            devices[name] = {'name': name, 'is_input': name in input_names, 'is_output': name in output_names}
    return list(devices.values())