import re
import ssl
import urllib.parse
import dogpile.cache
from dogpile.cache import api
from dogpile.cache import proxy
from dogpile.cache import util
from oslo_log import log
from oslo_utils import importutils
from oslo_cache._i18n import _
from oslo_cache import _opts
from oslo_cache import exception
def configure_cache_region(conf, region):
    """Configure a cache region.

    If the cache region is already configured, this function does nothing.
    Otherwise, the region is configured.

    :param conf: config object, must have had :func:`configure` called on it.
    :type conf: oslo_config.cfg.ConfigOpts
    :param region: Cache region to configure (see :func:`create_region`).
    :type region: dogpile.cache.region.CacheRegion
    :raises oslo_cache.exception.ConfigurationError: If the region parameter is
        not a dogpile.cache.CacheRegion.
    :returns: The region.
    :rtype: :class:`dogpile.cache.region.CacheRegion`
    """
    if not isinstance(region, dogpile.cache.CacheRegion):
        raise exception.ConfigurationError(_('region not type dogpile.cache.CacheRegion'))
    if not region.is_configured:
        config_dict = _build_cache_config(conf)
        region.configure_from_config(config_dict, '%s.' % conf.cache.config_prefix)
        if conf.cache.debug_cache_backend:
            region.wrap(_DebugProxy)
        if region.key_mangler is None:
            region.key_mangler = _sha1_mangle_key
        for class_path in conf.cache.proxies:
            cls = importutils.import_class(class_path)
            _LOG.debug("Adding cache-proxy '%s' to backend.", class_path)
            region.wrap(cls)
    return region