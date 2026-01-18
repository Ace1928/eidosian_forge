from __future__ import absolute_import, division, print_function
import json
import os
import shutil
from ansible.module_utils.six import raise_from
def compare_systemd_file_content(file_path, file_content):
    if not os.path.exists(file_path):
        return ('', file_content)
    with open(file_path, 'r') as unit_file:
        current_unit_file_content = unit_file.read()

    def remove_comments(content):
        return '\n'.join([line for line in content.splitlines() if not line.startswith('#')])
    current_unit_file_content_nocmnt = remove_comments(current_unit_file_content)
    unit_content_nocmnt = remove_comments(file_content)
    if current_unit_file_content_nocmnt == unit_content_nocmnt:
        return None
    diff_in_file = [line for line in unit_content_nocmnt.splitlines() if line not in current_unit_file_content_nocmnt.splitlines()]
    diff_in_string = [line for line in current_unit_file_content_nocmnt.splitlines() if line not in unit_content_nocmnt.splitlines()]
    return (diff_in_string, diff_in_file)