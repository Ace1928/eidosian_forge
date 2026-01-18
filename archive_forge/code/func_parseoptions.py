import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def parseoptions(self, args):
    """Parse command line options"""
    Options.location = args[0]
    del args[0]
    parser = CommandLineParser(Options)
    result = parser.parseoptions(args)
    if result:
        Trace.error(result)
        self.usage()
    self.processoptions()