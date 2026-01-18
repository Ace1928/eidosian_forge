import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parseupgreek(self, command, pos):
    """Parse the Greek \\up command.."""
    if len(command) < 4:
        return None
    if command.startswith('\\up'):
        upcommand = '\\' + command[3:]
    elif pos.checkskip('\\Up'):
        upcommand = '\\' + command[3:4].upper() + command[4:]
    else:
        Trace.error('Impossible upgreek command: ' + command)
        return
    upgreek = self.parsewithcommand(upcommand, pos)
    if upgreek:
        upgreek.type = 'font'
    return upgreek