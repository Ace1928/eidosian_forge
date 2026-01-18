from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def register_lazy_hook(self, hook_module_name, hook_member_name, hook_factory_member_name):
    self.register_lazy((hook_module_name, hook_member_name), hook_module_name, hook_factory_member_name)