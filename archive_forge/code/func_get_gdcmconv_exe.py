import os
import sys
import logging
import subprocess
from ..core import Format, BaseProgressIndicator, StdoutProgressIndicator
from ..core import read_n_bytes
def get_gdcmconv_exe():
    fname = 'gdcmconv' + '.exe' * sys.platform.startswith('win')
    try:
        subprocess.check_call([fname, '--version'])
        return [fname, '--raw']
    except Exception:
        pass
    candidates = []
    base_dirs = ['c:\\Program Files']
    for base_dir in base_dirs:
        if os.path.isdir(base_dir):
            for dname in os.listdir(base_dir):
                if dname.lower().startswith('gdcm'):
                    suffix = dname[4:].strip()
                    candidates.append((suffix, os.path.join(base_dir, dname)))
    candidates.sort(reverse=True)
    filename = None
    for _, dirname in candidates:
        exe1 = os.path.join(dirname, 'gdcmconv.exe')
        exe2 = os.path.join(dirname, 'bin', 'gdcmconv.exe')
        if os.path.isfile(exe1):
            filename = exe1
            break
        if os.path.isfile(exe2):
            filename = exe2
            break
    else:
        return None
    return [filename, '--raw']