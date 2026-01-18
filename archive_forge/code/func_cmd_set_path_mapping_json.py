import json
import os
import sys
import traceback
from _pydev_bundle import pydev_log
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydevd_bundle import pydevd_traceproperty, pydevd_dont_trace, pydevd_utils
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_breakpoints import get_exception_class
from _pydevd_bundle.pydevd_comm import (
from _pydevd_bundle.pydevd_constants import NEXT_VALUE_SEPARATOR, IS_WINDOWS, NULL
from _pydevd_bundle.pydevd_comm_constants import ID_TO_MEANING, CMD_EXEC_EXPRESSION, CMD_AUTHENTICATE
from _pydevd_bundle.pydevd_api import PyDevdAPI
from io import StringIO
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id
import pydevd_file_utils
def cmd_set_path_mapping_json(self, py_db, cmd_id, seq, text):
    """
        :param text:
            Json text. Something as:

            {
                "pathMappings": [
                    {
                        "localRoot": "c:/temp",
                        "remoteRoot": "/usr/temp"
                    }
                ],
                "debug": true,
                "force": false
            }
        """
    as_json = json.loads(text)
    force = as_json.get('force', False)
    path_mappings = []
    for pathMapping in as_json.get('pathMappings', []):
        localRoot = pathMapping.get('localRoot', '')
        remoteRoot = pathMapping.get('remoteRoot', '')
        if localRoot != '' and remoteRoot != '':
            path_mappings.append((localRoot, remoteRoot))
    if bool(path_mappings) or force:
        pydevd_file_utils.setup_client_server_paths(path_mappings)
    debug = as_json.get('debug', False)
    if debug or force:
        pydevd_file_utils.DEBUG_CLIENT_SERVER_TRANSLATION = debug