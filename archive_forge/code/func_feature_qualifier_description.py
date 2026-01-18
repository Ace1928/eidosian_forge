import re
import warnings
from Bio import BiopythonParserWarning
from Bio.Seq import Seq
from Bio.SeqFeature import Location
from Bio.SeqFeature import Reference
from Bio.SeqFeature import SeqFeature
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import LocationParserError
from .utils import FeatureValueCleaner
from .Scanner import GenBankScanner
def feature_qualifier_description(self, content):
    if '=' not in self._cur_qualifier.key:
        self._cur_qualifier.key = f'{self._cur_qualifier.key}='
    cur_content = self._remove_newlines(content)
    for remove_space_key in self.__class__.remove_space_keys:
        if remove_space_key in self._cur_qualifier.key:
            cur_content = self._remove_spaces(cur_content)
    self._cur_qualifier.value = self._normalize_spaces(cur_content)