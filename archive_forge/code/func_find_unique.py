import copy
from oslo_serialization import jsonutils
from urllib import parse
from saharaclient._i18n import _
def find_unique(self, **kwargs):
    found = self.find(**kwargs)
    if not found:
        raise APIException(error_code=404, error_message=_('No matches found.'))
    if len(found) > 1:
        raise APIException(error_code=409, error_message=_('Multiple matches found.'))
    return found[0]