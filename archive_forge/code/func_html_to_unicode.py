import re
import codecs
import encodings
from typing import Callable, Match, Optional, Tuple, Union, cast
from w3lib._types import AnyUnicodeError, StrOrBytes
import w3lib.util
def html_to_unicode(content_type_header: Optional[str], html_body_str: bytes, default_encoding: str='utf8', auto_detect_fun: Optional[Callable[[bytes], Optional[str]]]=None) -> Tuple[str, str]:
    '''Convert raw html bytes to unicode

    This attempts to make a reasonable guess at the content encoding of the
    html body, following a similar process to a web browser.

    It will try in order:

    * BOM (byte-order mark)
    * http content type header
    * meta or xml tag declarations
    * auto-detection, if the `auto_detect_fun` keyword argument is not ``None``
    * default encoding in keyword arg (which defaults to utf8)

    If an encoding other than the auto-detected or default encoding is used,
    overrides will be applied, converting some character encodings to more
    suitable alternatives.

    If a BOM is found matching the encoding, it will be stripped.

    The `auto_detect_fun` argument can be used to pass a function that will
    sniff the encoding of the text. This function must take the raw text as an
    argument and return the name of an encoding that python can process, or
    None.  To use chardet, for example, you can define the function as::

        auto_detect_fun=lambda x: chardet.detect(x).get('encoding')

    or to use UnicodeDammit (shipped with the BeautifulSoup library)::

        auto_detect_fun=lambda x: UnicodeDammit(x).originalEncoding

    If the locale of the website or user language preference is known, then a
    better default encoding can be supplied.

    If `content_type_header` is not present, ``None`` can be passed signifying
    that the header was not present.

    This method will not fail, if characters cannot be converted to unicode,
    ``\\\\ufffd`` (the unicode replacement character) will be inserted instead.

    Returns a tuple of ``(<encoding used>, <unicode_string>)``

    Examples:

    >>> import w3lib.encoding
    >>> w3lib.encoding.html_to_unicode(None,
    ... b"""<!DOCTYPE html>
    ... <head>
    ... <meta charset="UTF-8" />
    ... <meta name="viewport" content="width=device-width" />
    ... <title>Creative Commons France</title>
    ... <link rel='canonical' href='http://creativecommons.fr/' />
    ... <body>
    ... <p>Creative Commons est une organisation \\xc3\\xa0 but non lucratif
    ... qui a pour dessein de faciliter la diffusion et le partage des oeuvres
    ... tout en accompagnant les nouvelles pratiques de cr\\xc3\\xa9ation \\xc3\\xa0 l\\xe2\\x80\\x99\\xc3\\xa8re numerique.</p>
    ... </body>
    ... </html>""")
    ('utf-8', '<!DOCTYPE html>\\n<head>\\n<meta charset="UTF-8" />\\n<meta name="viewport" content="width=device-width" />\\n<title>Creative Commons France</title>\\n<link rel=\\'canonical\\' href=\\'http://creativecommons.fr/\\' />\\n<body>\\n<p>Creative Commons est une organisation \\xe0 but non lucratif\\nqui a pour dessein de faciliter la diffusion et le partage des oeuvres\\ntout en accompagnant les nouvelles pratiques de cr\\xe9ation \\xe0 l\\u2019\\xe8re numerique.</p>\\n</body>\\n</html>')
    >>>

    '''
    bom_enc, bom = read_bom(html_body_str)
    if bom_enc is not None:
        bom = cast(bytes, bom)
        return (bom_enc, to_unicode(html_body_str[len(bom):], bom_enc))
    enc = http_content_type_encoding(content_type_header)
    if enc is not None:
        if enc == 'utf-16' or enc == 'utf-32':
            enc += '-be'
        return (enc, to_unicode(html_body_str, enc))
    enc = html_body_declared_encoding(html_body_str)
    if enc is None and auto_detect_fun is not None:
        enc = auto_detect_fun(html_body_str)
    if enc is None:
        enc = default_encoding
    return (enc, to_unicode(html_body_str, enc))