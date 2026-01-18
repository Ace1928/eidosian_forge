import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def split_field_specifiers_iter(self, text):
    pos1 = 0
    while True:
        mo = ODFTranslator.field_pat.search(text, pos1)
        if mo:
            pos2 = mo.start()
            if pos2 > pos1:
                yield (ODFTranslator.code_text, text[pos1:pos2])
            yield (ODFTranslator.code_field, mo.group(1))
            pos1 = mo.end()
        else:
            break
    trailing = text[pos1:]
    if trailing:
        yield (ODFTranslator.code_text, trailing)