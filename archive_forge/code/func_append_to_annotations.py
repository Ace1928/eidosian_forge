from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def append_to_annotations(key, value):
    if key not in self.ParsedSeqRecord.annotations:
        self.ParsedSeqRecord.annotations[key] = []
    if value not in self.ParsedSeqRecord.annotations[key]:
        self.ParsedSeqRecord.annotations[key].append(value)