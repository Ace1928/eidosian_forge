import re
import string
def speedup(md):
    """Increase the speed of parsing paragraph and inline text."""
    md.block.register('paragraph', PARAGRAPH, parse_paragraph)
    punc = '\\\\><!\\[_*`~\\^\\$='
    text_pattern = '[\\s\\S]+?(?=[' + punc + ']|'
    if 'url_link' in md.inline.rules:
        text_pattern += 'https?:|'
    if md.inline.hard_wrap:
        text_pattern += ' *\\n|'
    else:
        text_pattern += ' {2,}\\n|'
    text_pattern += '$)'
    md.inline.register('text', text_pattern, parse_text)