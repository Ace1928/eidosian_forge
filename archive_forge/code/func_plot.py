import packages is to use `importr()`, for example
import rpy2.robjects as robjects
import rpy2.robjects.constants
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
from rpy2.robjects import rl
import warnings
def plot(self, vp=rpy2.robjects.constants.NULL):
    self._rprint(self, vp=vp)