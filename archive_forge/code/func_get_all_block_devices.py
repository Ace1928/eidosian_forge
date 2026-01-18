from __future__ import annotations
import errno
import os
def get_all_block_devices(self) -> list[str]:
    """Get the list of all block devices seen in /dev/disk/by-path/."""
    dir = '/dev/disk/by-path/'
    try:
        files = os.listdir(dir)
    except OSError as e:
        if e.errno == errno.ENOENT:
            files = []
        else:
            raise
    devices = []
    for file in files:
        devices.append(dir + file)
    return devices