import os
import re
import sys
def dos2unix_dir(dir_name):
    modified_files = []
    os.path.walk(dir_name, dos2unix_one_dir, modified_files)
    return modified_files