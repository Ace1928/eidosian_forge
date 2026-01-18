import re
import sys
import time
import logging
import platform
import threading
import subprocess as sp
import imageio_ffmpeg
import numpy as np
from ..core import Format, image_as_uint
def parse_device_names(ffmpeg_output):
    """Parse the output of the ffmpeg -list-devices command"""
    device_names = []
    in_video_devices = False
    for line in ffmpeg_output.splitlines():
        if line.startswith('[dshow'):
            logger.debug(line)
            line = line.split(']', 1)[1].strip()
            if in_video_devices and line.startswith('"'):
                friendly_name = line[1:-1]
                device_names.append([friendly_name, ''])
            elif in_video_devices and line.lower().startswith('alternative name'):
                alt_name = line.split(' name ', 1)[1].strip()[1:-1]
                if sys.platform.startswith('win'):
                    alt_name = alt_name.replace('&', '^&')
                else:
                    alt_name = alt_name.replace('&', '\\&')
                device_names[-1][-1] = alt_name
            elif 'video devices' in line:
                in_video_devices = True
            elif 'devices' in line:
                in_video_devices = False
    device_names2 = []
    for friendly_name, alt_name in device_names:
        if friendly_name not in device_names2:
            device_names2.append(friendly_name)
        elif alt_name:
            device_names2.append(alt_name)
        else:
            device_names2.append(friendly_name)
    return device_names2