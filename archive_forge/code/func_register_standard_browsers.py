import os
import shlex
import shutil
import sys
import subprocess
import threading
import warnings
def register_standard_browsers():
    global _tryorder
    _tryorder = []
    if sys.platform == 'darwin':
        register('MacOSX', None, MacOSXOSAScript('default'))
        register('chrome', None, MacOSXOSAScript('chrome'))
        register('firefox', None, MacOSXOSAScript('firefox'))
        register('safari', None, MacOSXOSAScript('safari'))
    if sys.platform == 'serenityos':
        register('Browser', None, BackgroundBrowser('Browser'))
    if sys.platform[:3] == 'win':
        register('windows-default', WindowsDefault)
        iexplore = os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'Internet Explorer\\IEXPLORE.EXE')
        for browser in ('firefox', 'firebird', 'seamonkey', 'mozilla', 'netscape', 'opera', iexplore):
            if shutil.which(browser):
                register(browser, None, BackgroundBrowser(browser))
    else:
        if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            try:
                cmd = 'xdg-settings get default-web-browser'.split()
                raw_result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
                result = raw_result.decode().strip()
            except (FileNotFoundError, subprocess.CalledProcessError, PermissionError, NotADirectoryError):
                pass
            else:
                global _os_preferred_browser
                _os_preferred_browser = result
            register_X_browsers()
        if os.environ.get('TERM'):
            if shutil.which('www-browser'):
                register('www-browser', None, GenericBrowser('www-browser'))
            if shutil.which('links'):
                register('links', None, GenericBrowser('links'))
            if shutil.which('elinks'):
                register('elinks', None, Elinks('elinks'))
            if shutil.which('lynx'):
                register('lynx', None, GenericBrowser('lynx'))
            if shutil.which('w3m'):
                register('w3m', None, GenericBrowser('w3m'))
    if 'BROWSER' in os.environ:
        userchoices = os.environ['BROWSER'].split(os.pathsep)
        userchoices.reverse()
        for cmdline in userchoices:
            if cmdline != '':
                cmd = _synthesize(cmdline, preferred=True)
                if cmd[1] is None:
                    register(cmdline, None, GenericBrowser(cmdline), preferred=True)