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
def record_end(self, content):
    """Signal the end of the record and do any necessary clean-up."""
    self.data.sequence = ''.join(self._seq_data)
    self._add_feature()