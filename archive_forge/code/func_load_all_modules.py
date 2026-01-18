import sys
from PyQt5 import QtWidgets
from DAWModules import SoundModule
from PyQt5.QtCore import Qt
import pyaudio
import numpy as np
from typing import Dict, Type, List, Tuple
def load_all_modules(self) -> None:
    """
        Dynamically loads all modules from the DAWModules.py file, registering each one.

        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the module does not conform to the expected interface.
        """
    import os
    import importlib
    module_dir = os.path.dirname(__file__)
    module_files = [f for f in os.listdir(module_dir) if f.endswith('.py') and f == 'DAWModules.py']
    for module_file in module_files:
        module_path = os.path.splitext(module_file)[0]
        module = importlib.import_module(module_path)
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isinstance(attribute, type) and issubclass(attribute, SoundModule) and (attribute is not SoundModule):
                try:
                    self.register_module(attribute_name, attribute)
                except ValueError as e:
                    print(f'Skipping registration for {attribute_name}: {e}')