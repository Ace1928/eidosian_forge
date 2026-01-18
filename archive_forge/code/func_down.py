import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
@classmethod
def down(cls, name, strict=False, recording=True):
    """ Return the number of Viewports it went down """
    cls._downviewport(name, strict=strict, recording=recording)