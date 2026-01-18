import enum
import shutil
import sys
def forceWrite(content, end=''):
    sys.stdout.write(str(content) + end)
    sys.stdout.flush()