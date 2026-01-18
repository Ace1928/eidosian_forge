from .. import osutils, tests, urlutils
from ..directory_service import directories
from ..location import hooks as location_hooks
from ..location import location_to_url, rcp_location_to_url
def get_base_location(self):
    path = osutils.abspath('/foo/bar')
    if path.startswith('/'):
        url = 'file://{}'.format(path)
    else:
        url = 'file:///{}'.format(path)
    return (path, url)