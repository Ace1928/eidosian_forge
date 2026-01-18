import re
from warnings import warn
from nltk.corpus import bcp47
def langname(tag, typ='full'):
    """
    Convert a composite BCP-47 tag to a language name

    >>> from nltk.langnames import langname
    >>> langname('ca-Latn-ES-valencia')
    'Catalan: Latin: Spain: Valencian'

    >>> langname('ca-Latn-ES-valencia', typ="short")
    'Catalan'
    """
    tags = tag.split('-')
    code = tags[0].lower()
    if codepattern.fullmatch(code):
        if code in iso639retired:
            return iso639retired[code]
        elif code in iso639short:
            code2 = iso639short[code]
            warn(f'Shortening {code!r} to {code2!r}', stacklevel=2)
            tag = '-'.join([code2] + tags[1:])
        name = bcp47.name(tag)
        if typ == 'full':
            return name
        elif name:
            return name.split(':')[0]
    else:
        warn(f'Could not find code in {code!r}', stacklevel=2)