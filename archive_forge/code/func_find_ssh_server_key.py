from copy import copy
import logging
import os.path
import sys
import paramiko
from os_ken import version
from os_ken.lib import hub
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
from os_ken.services.protocols.bgp.operator.commands.root import RootCmd
from os_ken.services.protocols.bgp.operator.internal_api import InternalApi
def find_ssh_server_key():
    if CONF[SSH_HOST_KEY]:
        return paramiko.RSAKey.from_private_key_file(CONF[SSH_HOST_KEY])
    elif os.path.exists('/etc/ssh_host_rsa_key'):
        return paramiko.RSAKey.from_private_key_file('/etc/ssh_host_rsa_key')
    elif os.path.exists('/etc/ssh/ssh_host_rsa_key'):
        return paramiko.RSAKey.from_private_key_file('/etc/ssh/ssh_host_rsa_key')
    else:
        return paramiko.RSAKey.generate(1024)