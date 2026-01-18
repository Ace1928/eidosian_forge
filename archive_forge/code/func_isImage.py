import sys
import glob
from builtins import range
def isImage(input, allExtensions):
    for extension in allExtensions:
        if input[-len(extension):] == extension:
            return (True, input[:-len(extension)], extension)
    return (False, input, '')