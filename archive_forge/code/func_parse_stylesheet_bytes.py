from webencodings import UTF8, decode, lookup
from .parser import parse_stylesheet
def parse_stylesheet_bytes(css_bytes, protocol_encoding=None, environment_encoding=None, skip_comments=False, skip_whitespace=False):
    """Parse :diagram:`stylesheet` from bytes,
    determining the character encoding as web browsers do.

    This is used when reading a file or fetching a URL.
    The character encoding is determined from the initial bytes
    (a :abbr:`BOM (Byte Order Mark)` or a ``@charset`` rule)
    as well as the parameters. The ultimate fallback is UTF-8.

    :type css_bytes: :obj:`bytes`
    :param css_bytes: A CSS byte string.
    :type protocol_encoding: :obj:`str`
    :param protocol_encoding:
        The encoding label, if any, defined by HTTP or equivalent protocol.
        (e.g. via the ``charset`` parameter of the ``Content-Type`` header.)
    :type environment_encoding: :class:`webencodings.Encoding`
    :param environment_encoding:
        The `environment encoding`_, if any.
    :type skip_comments: :obj:`bool`
    :param skip_comments:
        Ignore CSS comments at the top-level of the stylesheet.
        If the input is a string, ignore all comments.
    :type skip_whitespace: :obj:`bool`
    :param skip_whitespace:
        Ignore whitespace at the top-level of the stylesheet.
        Whitespace is still preserved
        in the :attr:`~tinycss2.ast.QualifiedRule.prelude`
        and the :attr:`~tinycss2.ast.QualifiedRule.content` of rules.
    :returns:
        A ``(rules, encoding)`` tuple.

        * ``rules`` is a list of
          :class:`~tinycss2.ast.QualifiedRule`,
          :class:`~tinycss2.ast.AtRule`,
          :class:`~tinycss2.ast.Comment` (if ``skip_comments`` is false),
          :class:`~tinycss2.ast.WhitespaceToken`
          (if ``skip_whitespace`` is false),
          and :class:`~tinycss2.ast.ParseError` objects.
        * ``encoding`` is the :class:`webencodings.Encoding` object
          that was used.
          If ``rules`` contains an ``@import`` rule, this is
          the `environment encoding`_ for the imported stylesheet.

    .. _environment encoding:
            https://www.w3.org/TR/css-syntax/#environment-encoding

    .. code-block:: python

        response = urlopen('http://example.net/foo.css')
        rules, encoding = parse_stylesheet_bytes(
            css_bytes=response.read(),
            # Python 3.x
            protocol_encoding=response.info().get_content_type().get_param('charset'),
            # Python 2.x
            protocol_encoding=response.info().gettype().getparam('charset'),
        )
        for rule in rules:
            ...

    """
    css_unicode, encoding = decode_stylesheet_bytes(css_bytes, protocol_encoding, environment_encoding)
    stylesheet = parse_stylesheet(css_unicode, skip_comments, skip_whitespace)
    return (stylesheet, encoding)