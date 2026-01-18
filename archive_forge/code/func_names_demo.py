import subprocess
import sys
from nltk.internals import find_binary
def names_demo():
    from nltk.classify.maxent import TadmMaxentClassifier
    from nltk.classify.util import names_demo
    classifier = names_demo(TadmMaxentClassifier.train)