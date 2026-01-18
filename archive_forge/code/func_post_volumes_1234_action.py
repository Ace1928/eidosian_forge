from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_volumes_1234_action(self, body, **kw):
    _body = None
    resp = 202
    assert len(list(body)) == 1
    action = list(body)[0]
    if action == 'os-attach':
        keys = sorted(list(body[action]))
        assert keys == ['instance_uuid', 'mode', 'mountpoint'] or keys == ['host_name', 'mode', 'mountpoint']
    elif action == 'os-detach':
        assert list(body[action]) == ['attachment_id']
    elif action == 'os-reserve':
        assert body[action] is None
    elif action == 'os-unreserve':
        assert body[action] is None
    elif action == 'os-initialize_connection':
        assert list(body[action]) == ['connector']
        return (202, {}, {'connection_info': {'foos': 'bars'}})
    elif action == 'os-terminate_connection':
        assert list(body[action]) == ['connector']
    elif action == 'os-begin_detaching':
        assert body[action] is None
    elif action == 'os-roll_detaching':
        assert body[action] is None
    elif action == 'os-reset_status':
        assert 'status' or 'attach_status' or 'migration_status' in body[action]
    elif action == 'os-extend':
        assert list(body[action]) == ['new_size']
    elif action == 'os-migrate_volume':
        assert 'host' in body[action]
        assert 'force_host_copy' in body[action]
    elif action == 'os-update_readonly_flag':
        assert list(body[action]) == ['readonly']
    elif action == 'os-retype':
        assert 'new_type' in body[action]
    elif action == 'os-set_bootable':
        assert list(body[action]) == ['bootable']
    elif action == 'os-unmanage':
        assert body[action] is None
    elif action == 'os-set_image_metadata':
        assert list(body[action]) == ['metadata']
    elif action == 'os-unset_image_metadata':
        assert 'key' in body[action]
    elif action == 'os-show_image_metadata':
        assert body[action] is None
    elif action == 'os-volume_upload_image':
        assert 'image_name' in body[action]
        _body = body
    elif action == 'revert':
        assert 'snapshot_id' in body[action]
    elif action == 'os-reimage':
        assert 'image_id' in body[action]
    elif action == 'os-extend_volume_completion':
        assert 'error' in body[action]
    else:
        raise AssertionError('Unexpected action: %s' % action)
    return (resp, {}, _body)