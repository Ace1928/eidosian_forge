import sys
from os import environ
def msgbox(message):
    if sys.platform == 'win32':
        import ctypes
        from ctypes.wintypes import LPCWSTR
        ctypes.windll.user32.MessageBoxW(None, LPCWSTR(message), u'Kivy Fatal Error', 0)
        sys.exit(1)