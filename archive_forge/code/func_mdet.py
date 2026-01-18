import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def mdet(self):
    return self.meigh[0].prod()