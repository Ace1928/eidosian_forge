from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat
@classmethod
def from_latex(cls, latex):
    import codecs
    import latexcodec
    from pybtex.markup import LaTeXParser
    return LaTeXParser(codecs.decode(latex, 'ulatex')).parse()