import tempfile, os
from pathlib import Path
from traitlets.config.loader import Config
def test_autorestore():
    ip.user_ns['foo'] = 95
    ip.magic('store foo')
    del ip.user_ns['foo']
    c = Config()
    c.StoreMagics.autorestore = False
    orig_config = ip.config
    try:
        ip.config = c
        ip.extension_manager.reload_extension('storemagic')
        assert 'foo' not in ip.user_ns
        c.StoreMagics.autorestore = True
        ip.extension_manager.reload_extension('storemagic')
        assert ip.user_ns['foo'] == 95
    finally:
        ip.config = orig_config