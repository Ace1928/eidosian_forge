import sys
from PyQt5 import QtWidgets
from DAWModules import SoundModule
from PyQt5.QtCore import Qt
import pyaudio
import numpy as np
from typing import Dict, Type, List, Tuple
def init_modules(self) -> Dict[str, SoundModule]:
    """
        Dynamically loads and initializes sound modules using the ModuleRegistry, handling failures gracefully and ensuring all modules are loaded if possible.

        Returns:
            Dict[str, SoundModule]: A dictionary of module names to their initialized instances.
        """
    module_registry = ModuleRegistry()
    module_registry.load_all_modules()
    modules = {}
    for module_name in module_registry.list_modules():
        module_class = module_registry.get_module(module_name)
        try:
            module_instance = module_class()
            modules[module_name] = module_instance
        except Exception as e:
            print(f'Failed to initialize module {module_name}: {e}')
    return modules