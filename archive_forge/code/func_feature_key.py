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
def feature_key(self, content):
    """Grab the key of the feature and signal the start of a new feature."""
    self._add_feature()
    from . import Record
    self._cur_feature = Record.Feature()
    self._cur_feature.key = content