import sys
from itertools import filterfalse
from typing import List, Tuple, Union
@_lazyclassproperty
def identchars(cls):
    """all characters in this range that are valid identifier characters, plus underscore '_'"""
    return ''.join(sorted(set(''.join(filter(str.isidentifier, cls._chars_for_ranges)) + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzªµº' + 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ' + '_')))