import tokenize
from six.moves import cStringIO as StringIO
from patsy import PatsyError
from patsy.origin import Origin
def pretty_untokenize(typed_tokens):
    text = []
    prev_was_space_delim = False
    prev_wants_space = False
    prev_was_open_paren_or_comma = False
    prev_was_object_like = False
    brackets = []
    for token_type, token in typed_tokens:
        assert token_type not in (tokenize.INDENT, tokenize.DEDENT, tokenize.NL)
        if token_type == tokenize.NEWLINE:
            continue
        if token_type == tokenize.ENDMARKER:
            continue
        if token_type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING):
            if prev_wants_space or prev_was_space_delim:
                text.append(' ')
            text.append(token)
            prev_wants_space = False
            prev_was_space_delim = True
        else:
            if token in ('(', '[', '{'):
                brackets.append(token)
            elif brackets and token in (')', ']', '}'):
                brackets.pop()
            this_wants_space_before = token in _python_space_before
            this_wants_space_after = token in _python_space_after
            if token == ':' and brackets and (brackets[-1] == '['):
                this_wants_space_after = False
            if token in ('*', '**') and prev_was_open_paren_or_comma:
                this_wants_space_before = False
                this_wants_space_after = False
            if token == '=' and (not brackets):
                this_wants_space_before = True
                this_wants_space_after = True
            if token in ('+', '-') and (not prev_was_object_like):
                this_wants_space_before = False
                this_wants_space_after = False
            if prev_wants_space or this_wants_space_before:
                text.append(' ')
            text.append(token)
            prev_wants_space = this_wants_space_after
            prev_was_space_delim = False
        if token_type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING) or token == ')':
            prev_was_object_like = True
        else:
            prev_was_object_like = False
        prev_was_open_paren_or_comma = token in ('(', ',')
    return ''.join(text)