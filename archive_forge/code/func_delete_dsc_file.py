import shutil
from oslo_utils import fileutils
import os_brick.privileged
@os_brick.privileged.default.entrypoint
def delete_dsc_file(file_name):
    return fileutils.delete_if_exists(file_name)