import os
import re
import sys
def dos2unix(file):
    """Replace CRLF with LF in argument files.  Print names of changed files."""
    if os.path.isdir(file):
        print(file, 'Directory!')
        return
    with open(file, 'rb') as fp:
        data = fp.read()
    if '\x00' in data:
        print(file, 'Binary!')
        return
    newdata = re.sub('\r\n', '\n', data)
    if newdata != data:
        print('dos2unix:', file)
        with open(file, 'wb') as f:
            f.write(newdata)
        return file
    else:
        print(file, 'ok')