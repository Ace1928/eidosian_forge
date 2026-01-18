import math
import sys
from Bio import MissingPythonDependencyError
def conf2str(conf):
    if int(conf) == conf:
        return str(int(conf))
    return str(conf)