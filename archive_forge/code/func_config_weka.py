import os
import re
import subprocess
import tempfile
import time
import zipfile
from sys import stdin
from nltk.classify.api import ClassifierI
from nltk.internals import config_java, java
from nltk.probability import DictionaryProbDist
def config_weka(classpath=None):
    global _weka_classpath
    config_java()
    if classpath is not None:
        _weka_classpath = classpath
    if _weka_classpath is None:
        searchpath = _weka_search
        if 'WEKAHOME' in os.environ:
            searchpath.insert(0, os.environ['WEKAHOME'])
        for path in searchpath:
            if os.path.exists(os.path.join(path, 'weka.jar')):
                _weka_classpath = os.path.join(path, 'weka.jar')
                version = _check_weka_version(_weka_classpath)
                if version:
                    print(f'[Found Weka: {_weka_classpath} (version {version})]')
                else:
                    print('[Found Weka: %s]' % _weka_classpath)
                _check_weka_version(_weka_classpath)
    if _weka_classpath is None:
        raise LookupError('Unable to find weka.jar!  Use config_weka() or set the WEKAHOME environment variable. For more information about Weka, please see https://www.cs.waikato.ac.nz/ml/weka/')