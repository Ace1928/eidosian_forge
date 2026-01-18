import logging
from cinderclient.apiclient import exceptions as apiclient_exception
from cinderclient import exceptions as cinder_exception
from keystoneauth1 import exceptions as keystone_exc
from oslo_utils import excutils
import retrying
from glance_store import exceptions
from glance_store.i18n import _LE
@handle_exceptions
def attachment_update(self, client, attachment_id, connector, mountpoint=None):
    """Updates the connector on the volume attachment. An attachment
        without a connector is considered reserved but not fully attached.

        :param client: cinderclient object
        :param attachment_id: UUID of the volume attachment to update.
        :param connector: host connector dict. This is required when updating
            a volume attachment. To terminate a connection, the volume
            attachment for that connection must be deleted.
        :param mountpoint: Optional mount device name for the attachment,
            e.g. "/dev/vdb". Theoretically this is optional per volume backend,
            but in practice it's normally required so it's best to always
            provide a value.
        :returns: a dict created from the
            cinderclient.v3.attachments.VolumeAttachment object with a backward
            compatible connection_info dict
        """
    if mountpoint and 'mountpoint' not in connector:
        connector['mountpoint'] = mountpoint
    try:
        attachment_ref = client.attachments.update(attachment_id, connector)
        return attachment_ref
    except cinder_exception.ClientException as ex:
        with excutils.save_and_reraise_exception():
            LOG.error(_LE('Update attachment failed for attachment %(id)s. Error: %(msg)s Code: %(code)s'), {'id': attachment_id, 'msg': str(ex), 'code': getattr(ex, 'code', None)})