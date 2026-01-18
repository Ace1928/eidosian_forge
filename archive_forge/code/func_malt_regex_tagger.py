import inspect
import os
import subprocess
import sys
import tempfile
from nltk.data import ZipFilePathPointer
from nltk.internals import find_dir, find_file, find_jars_within_path
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.parse.util import taggedsents_to_conll
def malt_regex_tagger():
    from nltk.tag import RegexpTagger
    _tagger = RegexpTagger([('\\.$', '.'), ('\\,$', ','), ('\\?$', '?'), ('\\($', '('), ('\\)$', ')'), ('\\[$', '['), ('\\]$', ']'), ('^-?[0-9]+(\\.[0-9]+)?$', 'CD'), ('(The|the|A|a|An|an)$', 'DT'), ('(He|he|She|she|It|it|I|me|Me|You|you)$', 'PRP'), ('(His|his|Her|her|Its|its)$', 'PRP$'), ('(my|Your|your|Yours|yours)$', 'PRP$'), ('(on|On|in|In|at|At|since|Since)$', 'IN'), ('(for|For|ago|Ago|before|Before)$', 'IN'), ('(till|Till|until|Until)$', 'IN'), ('(by|By|beside|Beside)$', 'IN'), ('(under|Under|below|Below)$', 'IN'), ('(over|Over|above|Above)$', 'IN'), ('(across|Across|through|Through)$', 'IN'), ('(into|Into|towards|Towards)$', 'IN'), ('(onto|Onto|from|From)$', 'IN'), ('.*able$', 'JJ'), ('.*ness$', 'NN'), ('.*ly$', 'RB'), ('.*s$', 'NNS'), ('.*ing$', 'VBG'), ('.*ed$', 'VBD'), ('.*', 'NN')])
    return _tagger.tag