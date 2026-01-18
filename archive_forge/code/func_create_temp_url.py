import copy
import uuid
from swiftclient import client as sc
from swiftclient import utils as swiftclient_utils
from urllib import parse as urlparse
from heatclient._i18n import _
from heatclient import exc
from heatclient.v1 import software_configs
def create_temp_url(swift_client, name, timeout, container=None):
    container = container or '%(name)s-%(uuid)s' % {'name': name, 'uuid': uuid.uuid4()}
    object_name = str(uuid.uuid4())
    swift_client.put_container(container)
    key_header = 'x-account-meta-temp-url-key'
    if key_header not in swift_client.head_account():
        swift_client.post_account({key_header: str(uuid.uuid4())[:32]})
    key = swift_client.head_account()[key_header]
    project_path = swift_client.url.split('/')[-1]
    path = '/v1/%s/%s/%s' % (project_path, container, object_name)
    timeout_secs = timeout * 60
    tempurl = swiftclient_utils.generate_temp_url(path, timeout_secs, key, 'PUT')
    sw_url = urlparse.urlparse(swift_client.url)
    put_url = '%s://%s%s' % (sw_url.scheme, sw_url.netloc, tempurl)
    swift_client.put_object(container, object_name, '')
    return put_url