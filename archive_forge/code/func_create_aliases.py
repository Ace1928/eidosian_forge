import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
def create_aliases():
    """Map common fonts that are absent from the system to similar fonts
    that are installed in the system
    """
    alias_groups = (('monospace', 'misc-fixed', 'courier', 'couriernew', 'console', 'fixed', 'mono', 'freemono', 'bitstreamverasansmono', 'verasansmono', 'monotype', 'lucidaconsole', 'consolas', 'dejavusansmono', 'liberationmono'), ('sans', 'arial', 'helvetica', 'swiss', 'freesans', 'bitstreamverasans', 'verasans', 'verdana', 'tahoma', 'calibri', 'gillsans', 'segoeui', 'trebuchetms', 'ubuntu', 'dejavusans', 'liberationsans'), ('serif', 'times', 'freeserif', 'bitstreamveraserif', 'roman', 'timesroman', 'timesnewroman', 'dutch', 'veraserif', 'georgia', 'cambria', 'constantia', 'dejavuserif', 'liberationserif'), ('wingdings', 'wingbats'), ('comicsansms', 'comicsans'))
    for alias_set in alias_groups:
        for name in alias_set:
            if name in Sysfonts:
                found = Sysfonts[name]
                break
        else:
            continue
        for name in alias_set:
            if name not in Sysfonts:
                Sysalias[name] = found