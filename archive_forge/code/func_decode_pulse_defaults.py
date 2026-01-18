from typing import Dict, Union, List
import dateutil.parser
def decode_pulse_defaults(defaults: Dict) -> None:
    """Decode pulse defaults data.

    Args:
        defaults: A ``PulseDefaults`` in dictionary format.
    """
    for item in defaults['pulse_library']:
        _decode_pulse_library_item(item)
    for cmd in defaults['cmd_def']:
        if 'sequence' in cmd:
            for instr in cmd['sequence']:
                _decode_pulse_qobj_instr(instr)