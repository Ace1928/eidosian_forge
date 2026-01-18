import re
import lxml
import lxml.etree
from lxml.html.clean import Cleaner
def etree_to_text(tree, guess_punct_space=True, guess_layout=True, newline_tags=NEWLINE_TAGS, double_newline_tags=DOUBLE_NEWLINE_TAGS):
    """
    Convert a html tree to text. Tree should be cleaned with
    ``html_text.html_text.cleaner.clean_html`` before passing to this
    function.

    See html_text.extract_text docstring for description of the
    approach and options.
    """
    chunks = []
    _NEWLINE = object()
    _DOUBLE_NEWLINE = object()

    class Context:
        """ workaround for missing `nonlocal` in Python 2 """
        prev = _DOUBLE_NEWLINE

    def should_add_space(text, prev):
        """ Return True if extra whitespace should be added before text """
        if prev in {_NEWLINE, _DOUBLE_NEWLINE}:
            return False
        if not guess_punct_space:
            return True
        if not _has_trailing_whitespace(prev):
            if _has_punct_after(text) or _has_open_bracket_before(prev):
                return False
        return True

    def get_space_between(text, prev):
        if not text:
            return ' '
        return ' ' if should_add_space(text, prev) else ''

    def add_newlines(tag, context):
        if not guess_layout:
            return
        prev = context.prev
        if prev is _DOUBLE_NEWLINE:
            return
        if tag in double_newline_tags:
            context.prev = _DOUBLE_NEWLINE
            chunks.append('\n' if prev is _NEWLINE else '\n\n')
        elif tag in newline_tags:
            context.prev = _NEWLINE
            if prev is not _NEWLINE:
                chunks.append('\n')

    def add_text(text_content, context):
        text = _normalize_whitespace(text_content) if text_content else ''
        if not text:
            return
        space = get_space_between(text, context.prev)
        chunks.extend([space, text])
        context.prev = text_content

    def traverse_text_fragments(tree, context, handle_tail=True):
        """ Extract text from the ``tree``: fill ``chunks`` variable """
        add_newlines(tree.tag, context)
        add_text(tree.text, context)
        for child in tree:
            traverse_text_fragments(child, context)
        add_newlines(tree.tag, context)
        if handle_tail:
            add_text(tree.tail, context)
    traverse_text_fragments(tree, context=Context(), handle_tail=False)
    return ''.join(chunks).strip()