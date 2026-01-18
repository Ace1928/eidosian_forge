import os
import shlex
import shutil
import sys
import subprocess
import threading
import warnings
def register_X_browsers():
    if shutil.which('xdg-open'):
        register('xdg-open', None, BackgroundBrowser('xdg-open'))
    if shutil.which('gio'):
        register('gio', None, BackgroundBrowser(['gio', 'open', '--', '%s']))
    if 'GNOME_DESKTOP_SESSION_ID' in os.environ and shutil.which('gvfs-open'):
        register('gvfs-open', None, BackgroundBrowser('gvfs-open'))
    if 'KDE_FULL_SESSION' in os.environ and shutil.which('kfmclient'):
        register('kfmclient', Konqueror, Konqueror('kfmclient'))
    if shutil.which('x-www-browser'):
        register('x-www-browser', None, BackgroundBrowser('x-www-browser'))
    for browser in ('firefox', 'iceweasel', 'iceape', 'seamonkey'):
        if shutil.which(browser):
            register(browser, None, Mozilla(browser))
    for browser in ('mozilla-firefox', 'mozilla-firebird', 'firebird', 'mozilla', 'netscape'):
        if shutil.which(browser):
            register(browser, None, Netscape(browser))
    if shutil.which('kfm'):
        register('kfm', Konqueror, Konqueror('kfm'))
    elif shutil.which('konqueror'):
        register('konqueror', Konqueror, Konqueror('konqueror'))
    for browser in ('galeon', 'epiphany'):
        if shutil.which(browser):
            register(browser, None, Galeon(browser))
    if shutil.which('skipstone'):
        register('skipstone', None, BackgroundBrowser('skipstone'))
    for browser in ('google-chrome', 'chrome', 'chromium', 'chromium-browser'):
        if shutil.which(browser):
            register(browser, None, Chrome(browser))
    if shutil.which('opera'):
        register('opera', None, Opera('opera'))
    if shutil.which('mosaic'):
        register('mosaic', None, BackgroundBrowser('mosaic'))
    if shutil.which('grail'):
        register('grail', Grail, None)