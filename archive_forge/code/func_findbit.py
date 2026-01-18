import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def findbit(self, piece):
    """Find the command bit corresponding to the \\begin{piece}"""
    for type in BeginCommand.types:
        if piece.replace('*', '') == type.piece:
            return self.factory.create(type)
    bit = self.factory.create(EquationEnvironment)
    bit.piece = piece
    return bit