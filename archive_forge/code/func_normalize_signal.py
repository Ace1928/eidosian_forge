from __future__ import absolute_import, division, print_function
import json
import os
import shutil
from ansible.module_utils.six import raise_from
def normalize_signal(signal_name_or_number):
    signal_name_or_number = str(signal_name_or_number)
    if signal_name_or_number.isdigit():
        return signal_name_or_number
    else:
        signal_name = signal_name_or_number.upper()
        if signal_name.startswith('SIG'):
            signal_name = signal_name[3:]
        if signal_name not in _signal_map:
            raise RuntimeError("Unknown signal '{0}'".format(signal_name_or_number))
        return str(_signal_map[signal_name])