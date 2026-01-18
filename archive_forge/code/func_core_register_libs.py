import os
import sysconfig
import sys
import traceback
import tempfile
import subprocess
import importlib
import kivy
from kivy.logger import Logger
def core_register_libs(category, libs, base='kivy.core'):
    if 'KIVY_DOC' in os.environ:
        return
    category = category.lower()
    kivy_options = kivy.kivy_options[category]
    libs_loadable = {}
    libs_ignored = []
    for option, lib in libs:
        if option not in kivy_options:
            Logger.debug('{0}: option <{1}> ignored by config'.format(category.capitalize(), option))
            libs_ignored.append(lib)
            continue
        libs_loadable[option] = lib
    libs_loaded = []
    for item in kivy_options:
        try:
            try:
                lib = libs_loadable[item]
            except KeyError:
                continue
            importlib.__import__(name='{2}.{0}.{1}'.format(category, lib, base), globals=globals(), locals=locals(), fromlist=[lib], level=0)
            libs_loaded.append(lib)
        except Exception as e:
            Logger.trace('{0}: Unable to use <{1}> as loader!'.format(category.capitalize(), option))
            Logger.trace('', exc_info=e)
            libs_ignored.append(lib)
    Logger.info('{0}: Providers: {1} {2}'.format(category.capitalize(), ', '.join(libs_loaded), '({0} ignored)'.format(', '.join(libs_ignored)) if libs_ignored else ''))
    return libs_loaded