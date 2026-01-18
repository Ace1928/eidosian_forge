import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def get_matching_blocks(self):
    size = min(len(self.b), len(self.b))
    threshold = min(self.threshold, size / 4)
    actual = difflib.SequenceMatcher.get_matching_blocks(self)
    return [item for item in actual if item[2] > threshold or not item[2]]