import csv
import gzip
import json
from nltk.internals import deprecated
@deprecated('Use open() and csv.writer() directly instead.')
def outf_writer_compat(outfile, encoding, errors, gzip_compress=False):
    """Get a CSV writer with optional compression."""
    return _outf_writer(outfile, encoding, errors, gzip_compress)