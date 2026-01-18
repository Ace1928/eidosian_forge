import os
import sys
import time
from rdkit import RDConfig
def isDebugBuild():
    try:
        return os.environ[BUILD_TYPE_ENVVAR] == 'DEBUG'
    except KeyError:
        return False