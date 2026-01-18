from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
@dbxrefs.deleter
def dbxrefs(self) -> None:
    del self._dbxrefs