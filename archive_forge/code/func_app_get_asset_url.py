from ._utils import AttributeDict
from . import exceptions
def app_get_asset_url(config, path):
    if config.assets_external_path:
        prefix = config.assets_external_path
    else:
        prefix = config.requests_pathname_prefix
    return '/'.join([prefix.rstrip('/'), config.assets_url_path.lstrip('/'), path])