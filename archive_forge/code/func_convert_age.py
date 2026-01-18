import re
from collections import defaultdict
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader
from nltk.util import LazyConcatenation, LazyMap, flatten
def convert_age(self, age_year):
    """Caclculate age in months from a string in CHILDES format"""
    m = re.match('P(\\d+)Y(\\d+)M?(\\d?\\d?)D?', age_year)
    age_month = int(m.group(1)) * 12 + int(m.group(2))
    try:
        if int(m.group(3)) > 15:
            age_month += 1
    except ValueError as e:
        pass
    return age_month