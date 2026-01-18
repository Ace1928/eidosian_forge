from html.parser import HTMLParser
def citation2latex(s):
    """Parse citations in Markdown cells.

    This looks for HTML tags having a data attribute names ``data-cite``
    and replaces it by the call to LaTeX cite command. The transformation
    looks like this::

        <cite data-cite="granger">(Granger, 2013)</cite>

    Becomes ::

        \\cite{granger}

    Any HTML tag can be used, which allows the citations to be formatted
    in HTML in any manner.
    """
    parser = CitationParser()
    parser.feed(s)
    parser.close()
    outtext = ''
    startpos = 0
    for citation in parser.citelist:
        outtext += s[startpos:citation[1]]
        outtext += '\\cite{%s}' % citation[0]
        startpos = citation[2] if len(citation) == 3 else -1
    outtext += s[startpos:] if startpos != -1 else ''
    return outtext