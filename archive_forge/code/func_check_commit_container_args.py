import base64
import binascii
import os
import re
import shlex
from oslo_serialization import jsonutils
from oslo_utils import netutils
from urllib import parse
from urllib import request
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import cliutils as utils
from zunclient import exceptions as exc
from zunclient.i18n import _
def check_commit_container_args(commit_args):
    opts = {}
    if commit_args.repository is not None:
        if ':' in commit_args.repository:
            args_list = commit_args.repository.rsplit(':')
            opts['repository'] = args_list[0]
            opts['tag'] = args_list[1]
        else:
            opts['repository'] = commit_args.repository
    return opts