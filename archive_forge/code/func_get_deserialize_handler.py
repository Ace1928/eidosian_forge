from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions as exception
def get_deserialize_handler(self, content_type):
    handlers = {'application/json': JSONDeserializer()}
    try:
        return handlers[content_type]
    except Exception:
        raise exception.InvalidContentType(content_type=content_type)