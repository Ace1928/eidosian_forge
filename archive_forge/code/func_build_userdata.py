import collections
import email
from email.mime import multipart
from email.mime import text
import os
import pkgutil
import string
from urllib import parse as urlparse
from neutronclient.common import exceptions as q_exceptions
from novaclient import api_versions
from novaclient import client as nc
from novaclient import exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients import microversion_mixin
from heat.engine.clients import os as os_client
from heat.engine import constraints
def build_userdata(self, metadata, userdata=None, instance_user=None, user_data_format='HEAT_CFNTOOLS'):
    """Build multipart data blob for CloudInit and Ignition.

        Data blob includes user-supplied Metadata, user data, and the required
        Heat in-instance configuration.

        :param resource: the resource implementation
        :type resource: heat.engine.Resource
        :param userdata: user data string
        :type userdata: str or None
        :param instance_user: the user to create on the server
        :type instance_user: string
        :param user_data_format: Format of user data to return
        :type user_data_format: string
        :returns: multipart mime as a string
        """
    if user_data_format == 'RAW':
        return userdata
    is_cfntools = user_data_format == 'HEAT_CFNTOOLS'
    is_software_config = user_data_format == 'SOFTWARE_CONFIG'
    if is_software_config and NovaClientPlugin.is_ignition_format(userdata):
        return NovaClientPlugin.build_ignition_data(metadata, userdata)

    def make_subpart(content, filename, subtype=None):
        if subtype is None:
            subtype = os.path.splitext(filename)[0]
        if content is None:
            content = ''
        try:
            content.encode('us-ascii')
            charset = 'us-ascii'
        except UnicodeEncodeError:
            charset = 'utf-8'
        msg = text.MIMEText(content, _subtype=subtype, _charset=charset) if subtype else text.MIMEText(content, _charset=charset)
        msg.add_header('Content-Disposition', 'attachment', filename=filename)
        return msg

    def read_cloudinit_file(fn):
        return pkgutil.get_data('heat', 'cloudinit/%s' % fn).decode('utf-8')
    if instance_user:
        config_custom_user = 'user: %s' % instance_user
        boothook_custom_user = "useradd -m %s\necho -e '%s\\tALL=(ALL)\\tNOPASSWD: ALL' >> /etc/sudoers\n" % (instance_user, instance_user)
    else:
        config_custom_user = ''
        boothook_custom_user = ''
    cloudinit_config = string.Template(read_cloudinit_file('config')).safe_substitute(add_custom_user=config_custom_user)
    cloudinit_boothook = string.Template(read_cloudinit_file('boothook.sh')).safe_substitute(add_custom_user=boothook_custom_user)
    attachments = [(cloudinit_config, 'cloud-config'), (cloudinit_boothook, 'boothook.sh', 'cloud-boothook'), (read_cloudinit_file('part_handler.py'), 'part-handler.py')]
    if is_cfntools:
        attachments.append((userdata, 'cfn-userdata', 'x-cfninitdata'))
    elif is_software_config:
        userdata_parts = None
        try:
            userdata_parts = email.message_from_string(userdata)
        except Exception:
            pass
        if userdata_parts and userdata_parts.is_multipart():
            for part in userdata_parts.get_payload():
                attachments.append((part.get_payload(), part.get_filename(), part.get_content_subtype()))
        else:
            attachments.append((userdata, ''))
    if is_cfntools:
        attachments.append((read_cloudinit_file('loguserdata.py'), 'loguserdata.py', 'x-shellscript'))
    if metadata:
        attachments.append((jsonutils.dumps(metadata), 'cfn-init-data', 'x-cfninitdata'))
    if is_cfntools:
        heat_client_plugin = self.context.clients.client_plugin('heat')
        cfn_md_url = heat_client_plugin.get_cfn_metadata_server_url()
        attachments.append((cfn_md_url, 'cfn-metadata-server', 'x-cfninitdata'))
        cfn_url = urlparse.urlparse(cfn_md_url)
        is_secure = cfg.CONF.instance_connection_is_secure
        vcerts = cfg.CONF.instance_connection_https_validate_certificates
        boto_cfg = '\n'.join(['[Boto]', 'debug = 0', 'is_secure = %s' % is_secure, 'https_validate_certificates = %s' % vcerts, 'cfn_region_name = heat', 'cfn_region_endpoint = %s' % cfn_url.hostname])
        attachments.append((boto_cfg, 'cfn-boto-cfg', 'x-cfninitdata'))
    subparts = [make_subpart(*args) for args in attachments]
    mime_blob = multipart.MIMEMultipart(_subparts=subparts)
    return mime_blob.as_string()