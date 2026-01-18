import os
import time
import base64
import binascii
from libcloud.utils import iso8601
from libcloud.utils.py3 import parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.storage.types import ObjectDoesNotExistError
from libcloud.common.azure_arm import AzureResourceManagementConnection
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import Provider
from libcloud.storage.drivers.azure_blobs import AzureBlobsStorageDriver
def ex_run_command(self, node, command, filerefs=[], timestamp=0, storage_account_name=None, storage_account_key=None, location=None):
    """
        Run a command on the node as root.

        Does not require ssh to log in,
        uses Windows Azure Agent (waagent) running
        on the node.

        :param node: The node on which to run the command.
        :type node: :class:``.Node``

        :param command: The actual command to run.  Note this is parsed
        into separate arguments according to shell quoting rules but is
        executed directly as a subprocess, not a shell command.
        :type command: ``str``

        :param filerefs: Optional files to fetch by URI from Azure blob store
        (must provide storage_account_name and storage_account_key),
        or regular HTTP.
        :type command: ``list`` of ``str``

        :param location: The location of the virtual machine
        (if None, use default location specified as 'region' in __init__)
        :type location: :class:`.NodeLocation`

        :param storage_account_name: The storage account
            from which to fetch files in `filerefs`
        :type storage_account_name: ``str``

        :param storage_account_key: The storage key to
            authorize to the blob store.
        :type storage_account_key: ``str``

        :type: ``list`` of :class:`.NodeLocation`

        """
    if location is None:
        if self.default_location:
            location = self.default_location
        else:
            raise ValueError('location is required.')
    name = 'init'
    target = node.id + '/extensions/' + name
    data = {'location': location.id, 'name': name, 'properties': {'publisher': 'Microsoft.OSTCExtensions', 'type': 'CustomScriptForLinux', 'typeHandlerVersion': '1.3', 'settings': {'fileUris': filerefs, 'commandToExecute': command, 'timestamp': timestamp}}}
    if storage_account_name and storage_account_key:
        data['properties']['protectedSettings'] = {'storageAccountName': storage_account_name, 'storageAccountKey': storage_account_key}
    r = self.connection.request(target, params={'api-version': VM_EXTENSION_API_VERSION}, data=data, method='PUT')
    return r.object