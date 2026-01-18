from fontTools.feaLib.error import FeatureLibError, IncludedFeaNotFound
from fontTools.feaLib.location import FeatureLibLocation
import re
import os
@staticmethod
def make_lexer_(file_or_path):
    if hasattr(file_or_path, 'read'):
        fileobj, closing = (file_or_path, False)
    else:
        filename, closing = (file_or_path, True)
        fileobj = open(filename, 'r', encoding='utf-8')
    data = fileobj.read()
    filename = getattr(fileobj, 'name', None)
    if closing:
        fileobj.close()
    return Lexer(data, filename)