from ironicclient.common import base
from ironicclient.common.i18n import _
from ironicclient import exc
def get_vendor_passthru_methods(self, driver_name, os_ironic_api_version=None, global_request_id=None):
    return self._get_as_dict('%s/vendor_passthru/methods' % driver_name, os_ironic_api_version=os_ironic_api_version, global_request_id=global_request_id)