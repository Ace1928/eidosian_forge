from ... import \
from ... import version_info  # noqa: F401
from ...hooks import install_lazy_named_hook
def register_lazy_option(key, member):
    config.option_registry.register_lazy(key, 'breezy.plugins.po_merge.po_merge', member)