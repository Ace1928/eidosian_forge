from osc_lib.command import command
from osc_lib import utils
from troveclient import exceptions
from troveclient.i18n import _
def set_attributes_for_print_detail(flavor):
    info = flavor._info.copy()
    if info.get('links'):
        del info['links']
    if hasattr(flavor, 'str_id'):
        info['id'] = flavor.id
        del info['str_id']
    return info