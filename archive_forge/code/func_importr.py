from rpy2.robjects.packages import importr as _importr
from rpy2.robjects.packages import data
import rpy2.robjects.help as rhelp
from rpy2.rinterface import baseenv
from os import linesep
from collections import OrderedDict
import re
def importr(packname, newname=None, verbose=False):
    """ Wrapper around rpy2.robjects.packages.importr, 
    adding the following feature(s):
    
    - package instance added to the pseudo-module 'packages'

    """
    assert isinstance(packname, str)
    packinstance = _importr(packname, on_conflict='warn')
    if newname is None:
        newname = packname.replace('.', '_')
    Packages().__dict__[newname] = packinstance
    return packinstance