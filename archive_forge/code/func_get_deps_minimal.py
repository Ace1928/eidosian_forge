imported modules that pyinstaller would not find on its own using
import os
import sys
import pkgutil
import logging
from os.path import dirname, join
import importlib
import subprocess
import re
import glob
import kivy
from kivy.factory import Factory
from PyInstaller.depend import bindepend
from os import environ
def get_deps_minimal(exclude_ignored=True, **kwargs):
    """Returns Kivy hidden modules as well as excluded modules to be used
    with ``Analysis``.

    The function takes core modules as keyword arguments and their value
    indicates which of the providers to include/exclude from the compiled app.

    The possible keyword names are ``audio, camera, clipboard, image, spelling,
    text, video, and window``. Their values can be:

        ``True``: Include current provider
            The providers imported when the core module is
            loaded on this system are added to hidden imports. This is the
            default if the keyword name is not specified.
        ``None``: Exclude
            Don't return this core module at all.
        ``A string or list of strings``: Providers to include
            Each string is the name of a provider for this module to be
            included.

    For example, ``get_deps_minimal(video=None, window=True,
    audio=['gstplayer', 'ffpyplayer'], spelling='enchant')`` will exclude all
    the video providers, will include the gstreamer and ffpyplayer providers
    for audio, will include the enchant provider for spelling, and will use the
    current default provider for ``window``.

    ``exclude_ignored``, if ``True`` (the default), if the value for a core
    library is ``None``, then if ``exclude_ignored`` is True, not only will the
    library not be included in the hiddenimports but it'll also added to the
    excluded imports to prevent it being included accidentally by pyinstaller.

    :returns:

        A dict with three keys, ``hiddenimports``, ``excludes``, and
        ``binaries``. Their values are a list of the corresponding modules to
        include/exclude. This can be passed directly to `Analysis`` with
        e.g. ::

            a = Analysis(['..\\kivy\\examples\\demo\\touchtracer\\main.py'],
                        ...
                         hookspath=hookspath(),
                         runtime_hooks=[],
                         win_no_prefer_redirects=False,
                         win_private_assemblies=False,
                         cipher=block_cipher,
                         **get_deps_minimal(video=None, audio=None))
    """
    core_mods = ['audio', 'camera', 'clipboard', 'image', 'spelling', 'text', 'video', 'window']
    mods = kivy_modules[:]
    excludes = excludedimports[:]
    for mod_name, val in kwargs.items():
        if mod_name not in core_mods:
            raise KeyError('{} not found in {}'.format(mod_name, core_mods))
        full_name = 'kivy.core.{}'.format(mod_name)
        if not val:
            core_mods.remove(mod_name)
            if exclude_ignored:
                excludes.extend(collect_submodules(full_name))
            continue
        if val is True:
            continue
        core_mods.remove(mod_name)
        mods.append(full_name)
        single_mod = False
        if isinstance(val, (str, bytes)):
            single_mod = True
            mods.append('kivy.core.{0}.{0}_{1}'.format(mod_name, val))
        if not single_mod:
            for v in val:
                mods.append('kivy.core.{0}.{0}_{1}'.format(mod_name, v))
    for mod_name in core_mods:
        full_name = 'kivy.core.{}'.format(mod_name)
        mods.append(full_name)
        m = importlib.import_module(full_name)
        if mod_name == 'clipboard' and m.CutBuffer:
            mods.append(m.CutBuffer.__module__)
        if hasattr(m, mod_name.capitalize()):
            val = getattr(m, mod_name.capitalize())
            if val:
                mods.append(getattr(val, '__module__'))
        if hasattr(m, 'libs_loaded') and m.libs_loaded:
            for name in m.libs_loaded:
                mods.append('kivy.core.{}.{}'.format(mod_name, name))
    mods = sorted(set(mods))
    binaries = []
    if any(('gstplayer' in m for m in mods)):
        binaries = _find_gst_binaries()
    elif exclude_ignored:
        excludes.append('kivy.lib.gstplayer')
    return {'hiddenimports': mods, 'excludes': excludes, 'binaries': binaries}