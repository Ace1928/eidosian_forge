from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from datetime import datetime, timedelta
import time
import copy
def check_snapshot_modified(snapshot=None, desired_retention=None, retention_unit=None, size=None, cap_unit=None, access_mode=None):
    """Check if snapshot modification is required
        :param snapshot: Snapshot details
        :param desired_retention: Desired retention of the snapshot
        :param retention_unit: Retention unit for snapshot
        :param size: Size of the snapshot
        :param cap_unit: Capacity unit for the snapshot
        :param access_mode: Access mode of the snapshot
        :return: Boolean indicating if modification is needed
    """
    snap_creation_timestamp = None
    expiration_timestamp = None
    is_timestamp_modified = False
    is_size_modified = False
    is_access_modified = False
    is_modified = False
    if 'creationTime' in snapshot:
        snap_creation_timestamp = snapshot['creationTime']
    if desired_retention:
        if retention_unit == 'hours':
            expiration_timestamp = datetime.fromtimestamp(snap_creation_timestamp) + timedelta(hours=desired_retention)
            expiration_timestamp = time.mktime(expiration_timestamp.timetuple())
        else:
            expiration_timestamp = datetime.fromtimestamp(snap_creation_timestamp) + timedelta(days=desired_retention)
            expiration_timestamp = time.mktime(expiration_timestamp.timetuple())
    if 'secureSnapshotExpTime' in snapshot and expiration_timestamp and (snapshot['secureSnapshotExpTime'] != expiration_timestamp):
        existing_timestamp = snapshot['secureSnapshotExpTime']
        new_timestamp = expiration_timestamp
        info_message = 'The existing timestamp is: %s and the new timestamp is: %s' % (existing_timestamp, new_timestamp)
        LOG.info(info_message)
        existing_time_obj = datetime.fromtimestamp(existing_timestamp)
        new_time_obj = datetime.fromtimestamp(new_timestamp)
        if existing_time_obj > new_time_obj:
            td = utils.dateutil.relativedelta.relativedelta(existing_time_obj, new_time_obj)
        else:
            td = utils.dateutil.relativedelta.relativedelta(new_time_obj, existing_time_obj)
        LOG.info('Time difference: %s', td.minutes)
        if td.seconds > 120 or td.minutes > 2:
            is_timestamp_modified = True
    if size:
        if cap_unit == 'GB':
            new_size = size * 1024 * 1024
        else:
            new_size = size * 1024 * 1024 * 1024
        if new_size != snapshot['sizeInKb']:
            is_size_modified = True
    if access_mode and snapshot['accessModeLimit'] != access_mode:
        is_access_modified = True
    if is_timestamp_modified or is_size_modified or is_access_modified:
        is_modified = True
    return (is_modified, is_timestamp_modified, is_size_modified, is_access_modified)