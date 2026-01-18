import dataclasses
from dataclasses import field
from types import CodeType, ModuleType
from typing import Any, Dict
def record_module_access(self, mod, name, val):
    if self._is_excl(mod):
        return
    if isinstance(val, ModuleType):
        self.name_to_modrec[mod.__name__].accessed_attrs[name] = self._add_mod(val)
        return
    if mod.__name__ in self.name_to_modrec:
        self.name_to_modrec[mod.__name__].accessed_attrs[name] = val