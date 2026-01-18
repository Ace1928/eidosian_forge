import logging
from cinderclient.apiclient import exceptions as apiclient_exception
from cinderclient import exceptions as cinder_exception
from keystoneauth1 import exceptions as keystone_exc
from oslo_utils import excutils
import retrying
from glance_store import exceptions
from glance_store.i18n import _LE
@handle_exceptions
def attachment_get(self, client, attachment_id):
    """Gets a volume attachment.

        :param client: cinderclient object
        :param attachment_id: UUID of the volume attachment to get.
        :returns: a dict created from the
            cinderclient.v3.attachments.VolumeAttachment object with a backward
            compatible connection_info dict
        """
    try:
        attachment_ref = client.attachments.show(attachment_id)
        return attachment_ref
    except cinder_exception.ClientException as ex:
        with excutils.save_and_reraise_exception():
            LOG.error(_LE('Show attachment failed for attachment %(id)s. Error: %(msg)s Code: %(code)s'), {'id': attachment_id, 'msg': str(ex), 'code': getattr(ex, 'code', None)})