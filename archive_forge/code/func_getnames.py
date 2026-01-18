from pathlib import Path
import sys
def getnames(self):
    """ Return a list of all member names in the archive. """
    return [f.name for f in self.__members]