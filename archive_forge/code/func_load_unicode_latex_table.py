import codecs
import dataclasses
import unicodedata
from typing import Optional, List, Union, Any, Iterator, Tuple, Type, Dict
from latexcodec import lexer
from codecs import CodecInfo
def load_unicode_latex_table() -> Iterator[UnicodeLatexTranslation]:
    with pkg_resources.open_text('latexcodec', 'table.txt') as datafile:
        for line in datafile:
            marker, unicode_names, latex = line.rstrip('\r\n').split('\t')
            unicode = ''.join((unicodedata.lookup(name) for name in unicode_names.split(',')))
            yield UnicodeLatexTranslation(unicode=unicode, latex=latex, encode=marker[1] in {'-', '>'}, decode=marker[1] in {'-', '<'}, text_mode=marker[0] in {'A', 'T'}, math_mode=marker[0] in {'A', 'M'})