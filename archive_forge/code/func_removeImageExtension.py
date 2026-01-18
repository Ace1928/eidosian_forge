import sys
import glob
from builtins import range
def removeImageExtension(input, allExtensions):
    return isImage(input, allExtensions)[1]