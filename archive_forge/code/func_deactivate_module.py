from kivy.config import Config
from kivy.logger import Logger
import kivy
import importlib
import os
import sys
def deactivate_module(self, name, win):
    """Deactivate a module from a window"""
    if name not in self.mods:
        Logger.warning('Modules: Module <%s> not found' % name)
        return
    if 'module' not in self.mods[name]:
        return
    module = self.mods[name]['module']
    if self.mods[name]['activated']:
        module.stop(win, self.mods[name]['context'])
        self.mods[name]['activated'] = False