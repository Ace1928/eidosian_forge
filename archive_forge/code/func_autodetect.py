from __future__ import absolute_import
import logging
import os
def autodetect():
    """Detects an appropriate cache module and returns it.

    Returns:
      googleapiclient.discovery_cache.base.Cache, a cache object which
      is auto detected, or None if no cache object is available.
    """
    if 'APPENGINE_RUNTIME' in os.environ:
        try:
            from . import appengine_memcache
            return appengine_memcache.cache
        except Exception:
            pass
    try:
        from . import file_cache
        return file_cache.cache
    except Exception:
        LOGGER.info('file_cache is only supported with oauth2client<4.0.0', exc_info=False)
        return None