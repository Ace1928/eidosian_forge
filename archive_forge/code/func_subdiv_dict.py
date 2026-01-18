import re
from warnings import warn
from xml.etree import ElementTree as et
from nltk.corpus.reader import CorpusReader
def subdiv_dict(self, subdivs):
    """Convert the CLDR subdivisions list to a dictionary"""
    return {sub.attrib['type']: sub.text for sub in subdivs}