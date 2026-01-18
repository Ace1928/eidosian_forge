from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def coercePubid(self, data):
    dataOutput = data
    for char in nonPubidCharRegexp.findall(data):
        warnings.warn('Coercing non-XML pubid', DataLossWarning)
        replacement = self.getReplacementCharacter(char)
        dataOutput = dataOutput.replace(char, replacement)
    if self.preventSingleQuotePubid and dataOutput.find("'") >= 0:
        warnings.warn('Pubid cannot contain single quote', DataLossWarning)
        dataOutput = dataOutput.replace("'", self.getReplacementCharacter("'"))
    return dataOutput