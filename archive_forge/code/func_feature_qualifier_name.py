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
def feature_qualifier_name(self, content_list):
    """Deal with qualifier names.

        We receive a list of keys, since you can have valueless keys such as
        /pseudo which would be passed in with the next key (since no other
        tags separate them in the file)
        """
    from . import Record
    for content in content_list:
        if not content.startswith('/'):
            content = f'/{content}'
        if self._cur_qualifier is not None:
            self._cur_feature.qualifiers.append(self._cur_qualifier)
        self._cur_qualifier = Record.Qualifier()
        self._cur_qualifier.key = content