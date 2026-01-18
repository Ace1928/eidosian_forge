from __future__ import absolute_import, division, print_function
from . import errors, http
def get_available_asset_versions(namespace, name):
    asset_data = get('{0}/{1}'.format(namespace, name))
    try:
        return set((v['version'] for v in asset_data['versions']))
    except (TypeError, KeyError):
        raise errors.BonsaiError('Cannot extract versions from {0}'.format(asset_data))