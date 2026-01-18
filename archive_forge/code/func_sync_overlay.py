from __future__ import absolute_import, division, print_function
import shutil
import traceback
from os import path
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
def sync_overlay(name):
    """Synchronizes the specified overlay repository.

    :param name: the overlay repository id to sync
    :raises ModuleError
    """
    layman = init_layman()
    if not layman.sync(name):
        messages = [str(item[1]) for item in layman.sync_results[2]]
        raise ModuleError(messages)