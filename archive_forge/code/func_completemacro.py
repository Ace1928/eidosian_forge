import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def completemacro(self, macro):
    """Complete the macro with the parameters read."""
    self.contents = [macro.instantiate()]
    replaced = [False] * len(self.values)
    for parameter in self.searchall(MacroParameter):
        index = parameter.number - 1
        if index >= len(self.values):
            Trace.error('Macro parameter index out of bounds: ' + str(index))
            return
        replaced[index] = True
        parameter.contents = [self.values[index].clone()]
    for index in range(len(self.values)):
        if not replaced[index]:
            self.addfilter(index, self.values[index])