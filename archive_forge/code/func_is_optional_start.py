from __future__ import absolute_import, division, unicode_literals
from . import base
def is_optional_start(self, tagname, previous, next):
    type = next and next['type'] or None
    if tagname in 'html':
        return type not in ('Comment', 'SpaceCharacters')
    elif tagname == 'head':
        if type in ('StartTag', 'EmptyTag'):
            return True
        elif type == 'EndTag':
            return next['name'] == 'head'
    elif tagname == 'body':
        if type in ('Comment', 'SpaceCharacters'):
            return False
        elif type == 'StartTag':
            return next['name'] not in ('script', 'style')
        else:
            return True
    elif tagname == 'colgroup':
        if type in ('StartTag', 'EmptyTag'):
            return next['name'] == 'col'
        else:
            return False
    elif tagname == 'tbody':
        if type == 'StartTag':
            if previous and previous['type'] == 'EndTag' and (previous['name'] in ('tbody', 'thead', 'tfoot')):
                return False
            return next['name'] == 'tr'
        else:
            return False
    return False