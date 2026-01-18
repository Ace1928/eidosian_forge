import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
def buffer(self):
    """Buffer the entire stream (can consume a considerable amount of
        memory).

        Useful in conjunction with copy(accumulate=True) and
        cut(accumulate=True) to ensure that all marked events in the entire
        stream are copied to the buffer before further transformations are
        applied.

        For example, to move all <note> elements inside a <notes> tag at the
        top of the document:

        >>> doc = HTML('<doc><notes></notes><body>Some <note>one</note> '
        ...            'text <note>two</note>.</body></doc>',
        ...             encoding='utf-8')
        >>> buffer = StreamBuffer()
        >>> print(doc | Transformer('body/note').cut(buffer, accumulate=True)
        ...     .end().buffer().select('notes').prepend(buffer))
        <doc><notes><note>one</note><note>two</note></notes><body>Some  text
        .</body></doc>

        """
    return self.apply(list)