import base64
import hashlib
import logging
import os
import re
import tempfile
import uuid
from collections import namedtuple
from rdkit import Chem, RDConfig
from rdkit.Chem.MolKey import InchiInfo
def initStruchk(configDir=None, logFile=None):
    global __initCalled
    if configDir is None:
        configDir = os.path.join(RDConfig.RDDataDir, 'struchk')
    if configDir[-1] != os.path.sep:
        configDir += os.path.sep
    if logFile is None:
        fd = tempfile.NamedTemporaryFile(suffix='.log', delete=False)
        fd.close()
        logFile = fd.name
    struchk_init = '-tm\n-ta {0}checkfgs.trn\n-tm\n-or\n-ca {0}checkfgs.chk\n-cc\n-cl 3\n-cs\n-cn 999\n-l {1}\n'.format(configDir, logFile)
    initRes = pyAvalonTools.InitializeCheckMol(struchk_init)
    if initRes:
        raise ValueError(f'bad result from InitializeCheckMol: {initRes}')
    __initCalled = True