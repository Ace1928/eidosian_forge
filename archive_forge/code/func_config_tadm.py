import subprocess
import sys
from nltk.internals import find_binary
def config_tadm(bin=None):
    global _tadm_bin
    _tadm_bin = find_binary('tadm', bin, env_vars=['TADM'], binary_names=['tadm'], url='http://tadm.sf.net')