from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
@fix_unicode_literals_in_doctest
def split_name_list(string):
    """
    Split a list of names, separated by ' and '.

    >>> split_name_list('Johnson and Peterson')
    [u'Johnson', u'Peterson']
    >>> split_name_list('Johnson AND Peterson')
    [u'Johnson', u'Peterson']
    >>> split_name_list('Johnson AnD Peterson')
    [u'Johnson', u'Peterson']
    >>> split_name_list('Armand and Peterson')
    [u'Armand', u'Peterson']
    >>> split_name_list('Armand and anderssen')
    [u'Armand', u'anderssen']
    >>> split_name_list('{Armand and Anderssen}')
    [u'{Armand and Anderssen}']
    >>> split_name_list('What a Strange{ }and Bizzare Name! and Peterson')
    [u'What a Strange{ }and Bizzare Name!', u'Peterson']
    >>> split_name_list('What a Strange and{ }Bizzare Name! and Peterson')
    [u'What a Strange and{ }Bizzare Name!', u'Peterson']
    """
    return split_tex_string(string, ' [Aa][Nn][Dd] ')