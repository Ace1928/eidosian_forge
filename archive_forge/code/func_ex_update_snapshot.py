import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_update_snapshot(self, perm_to_create_volume_addition_account_id: List[str]=None, perm_to_create_volume_addition_global_perm: bool=None, perm_to_create_volume_removals_account_id: List[str]=None, perm_to_create_volume_removals_global_perm: bool=None, snapshot: VolumeSnapshot=None, dry_run: bool=False):
    """
        Modifies the permissions for a specified snapshot.
        You can add or remove permissions for specified account IDs or groups.
        You can share a snapshot with a user that is in the same Region.
        The user can create a copy of the snapshot you shared, obtaining all
        the rights for the copy of the snapshot.

        :param      perm_to_create_volume_addition_account_id:
        The account ID of one or more users who have permissions
        for the resource.
        :type       perm_to_create_volume_addition_account_id:
        ``list`` of ``str``

        :param      perm_to_create_volume_addition_global_perm: If true,
        the resource is public. If false, the resource is private.
        :type       perm_to_create_volume_addition_global_perm: ``bool``

        :param      perm_to_create_volume_removals_account_id: The account ID
         of one or more users who have permissions for the resource.
        :type       perm_to_create_volume_removals_account_id:
        ``list`` of ``str``

        :param      perm_to_create_volume_removals_global_perm: If true,
         the resource is public. If false, the resource is private.
        :type       perm_to_create_volume_removals_global_perm: ``bool``

        :param      snapshot: The ID of the snapshot. (required)
        :type       snapshot: ``VolumeSnapshot``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: snapshot export tasks
        :rtype: ``list`` of ``dict``
        """
    action = 'UpdateSnapshot'
    data = {'DryRun: ': dry_run, 'PermissionsToCreateVolume': {'Additions': {}, 'Removals': {}}}
    if snapshot is not None:
        data.update({'ImageId': snapshot.id})
    if perm_to_create_volume_addition_account_id is not None:
        data['PermissionsToCreateVolume']['Additions'].update({'AccountIds': perm_to_create_volume_addition_account_id})
    if perm_to_create_volume_addition_global_perm is not None:
        data['PermissionsToCreateVolume']['Additions'].update({'GlobalPermission': perm_to_create_volume_addition_global_perm})
    if perm_to_create_volume_removals_account_id is not None:
        data['PermissionsToCreateVolume']['Removals'].update({'AccountIds': perm_to_create_volume_removals_account_id})
    if perm_to_create_volume_removals_global_perm is not None:
        data['PermissionsToCreateVolume']['Removals'].update({'GlobalPermission': perm_to_create_volume_removals_global_perm})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['Snapshot']
    return response.json()