from __future__ import absolute_import, division, print_function
from . import errors, http
def get_asset_parameters(name, version):
    try:
        namespace, asset_name = name.split('/')
    except ValueError:
        raise errors.BonsaiError('Bonsai asset names should be formatted as <namespace>/<name>.')
    available_versions = get_available_asset_versions(namespace, asset_name)
    if version not in available_versions:
        raise errors.BonsaiError('Version {0} is not available. Choose from: {1}.'.format(version, ', '.join(available_versions)))
    asset_builds = get_asset_version_builds(namespace, asset_name, version)
    return dict(labels=asset_builds.get('metadata', {}).get('labels'), annotations=asset_builds.get('metadata', {}).get('annotations'), builds=asset_builds['spec']['builds'])