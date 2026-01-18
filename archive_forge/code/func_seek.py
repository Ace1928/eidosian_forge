import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
@classmethod
def seek(cls, name, recording=True):
    """ Seek and return a Viewport given its name """
    cls._seek(name, recording=recording)