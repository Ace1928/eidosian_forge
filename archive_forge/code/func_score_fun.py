import pickle
import re
from debian.deprecation import function_deprecated_by
def score_fun(x):
    return float((x - 15) * (x - 15)) / x