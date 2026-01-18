import shutil
from oslo_utils import fileutils
import os_brick.privileged
@os_brick.privileged.default.entrypoint
def move_dsc_file(src, dst):
    return shutil.move(src, dst)