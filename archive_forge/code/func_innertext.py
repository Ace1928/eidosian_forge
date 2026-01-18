import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def innertext(self, pos):
    """Parse some text inside the bracket, following textual rules."""
    specialchars = list(FormulaConfig.symbolfunctions.keys())
    specialchars.append(FormulaConfig.starts['command'])
    specialchars.append(FormulaConfig.starts['bracket'])
    specialchars.append(Comment.start)
    while not pos.finished():
        if pos.current() in specialchars:
            self.add(self.factory.parseany(pos))
            if pos.checkskip(' '):
                self.original += ' '
        else:
            self.add(FormulaConstant(pos.skipcurrent()))