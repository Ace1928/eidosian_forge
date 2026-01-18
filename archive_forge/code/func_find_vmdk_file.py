from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def find_vmdk_file(self, datastore_obj, vmdk_fullpath, vmdk_filename, vmdk_folder):
    """
        Return vSphere file object or fail_json
        Args:
            datastore_obj: Managed object of datastore
            vmdk_fullpath: Path of VMDK file e.g., path/to/vm/vmdk_filename.vmdk
            vmdk_filename: Name of vmdk e.g., VM0001_1.vmdk
            vmdk_folder: Base dir of VMDK e.g, path/to/vm

        """
    browser = datastore_obj.browser
    datastore_name = datastore_obj.name
    datastore_name_sq = '[' + datastore_name + ']'
    if browser is None:
        self.module.fail_json(msg='Unable to access browser for datastore %s' % datastore_name)
    detail_query = vim.host.DatastoreBrowser.FileInfo.Details(fileOwner=True, fileSize=True, fileType=True, modification=True)
    search_spec = vim.host.DatastoreBrowser.SearchSpec(details=detail_query, matchPattern=[vmdk_filename], searchCaseInsensitive=True)
    search_res = browser.SearchSubFolders(datastorePath=datastore_name_sq, searchSpec=search_spec)
    changed = False
    vmdk_path = datastore_name_sq + ' ' + vmdk_fullpath
    try:
        changed, result = wait_for_task(search_res)
    except TaskError as task_e:
        self.module.fail_json(msg=to_native(task_e))
    if not changed:
        self.module.fail_json(msg='No valid disk vmdk image found for path %s' % vmdk_path)
    target_folder_paths = [datastore_name_sq + ' ' + vmdk_folder + '/', datastore_name_sq + ' ' + vmdk_folder]
    for file_result in search_res.info.result:
        for f in getattr(file_result, 'file'):
            if f.path == vmdk_filename and file_result.folderPath in target_folder_paths:
                return f
    self.module.fail_json(msg='No vmdk file found for path specified [%s]' % vmdk_path)