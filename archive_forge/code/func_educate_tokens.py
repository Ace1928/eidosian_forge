from a single quote by the algorithm. Therefore, a text like::
import re, sys
def educate_tokens(text_tokens, attr=default_smartypants_attr, language='en'):
    """Return iterator that "educates" the items of `text_tokens`.
    """
    convert_quot = False
    do_dashes = False
    do_backticks = False
    do_quotes = False
    do_ellipses = False
    do_stupefy = False
    if attr == '1':
        do_quotes = True
        do_backticks = True
        do_dashes = 1
        do_ellipses = True
    elif attr == '2':
        do_quotes = True
        do_backticks = True
        do_dashes = 2
        do_ellipses = True
    elif attr == '3':
        do_quotes = True
        do_backticks = True
        do_dashes = 3
        do_ellipses = True
    elif attr == '-1':
        do_stupefy = True
    else:
        if 'q' in attr:
            do_quotes = True
        if 'b' in attr:
            do_backticks = True
        if 'B' in attr:
            do_backticks = 2
        if 'd' in attr:
            do_dashes = 1
        if 'D' in attr:
            do_dashes = 2
        if 'i' in attr:
            do_dashes = 3
        if 'e' in attr:
            do_ellipses = True
        if 'w' in attr:
            convert_quot = True
    prev_token_last_char = ' '
    for ttype, text in text_tokens:
        if ttype == 'tag' or not text:
            yield text
            continue
        if ttype == 'literal':
            prev_token_last_char = text[-1:]
            yield text
            continue
        last_char = text[-1:]
        text = processEscapes(text)
        if convert_quot:
            text = re.sub('&quot;', '"', text)
        if do_dashes == 1:
            text = educateDashes(text)
        elif do_dashes == 2:
            text = educateDashesOldSchool(text)
        elif do_dashes == 3:
            text = educateDashesOldSchoolInverted(text)
        if do_ellipses:
            text = educateEllipses(text)
        if do_backticks:
            text = educateBackticks(text, language)
        if do_backticks == 2:
            text = educateSingleBackticks(text, language)
        if do_quotes:
            context = prev_token_last_char.replace('"', ';').replace("'", ';')
            text = educateQuotes(context + text, language)[1:]
        if do_stupefy:
            text = stupefyEntities(text, language)
        prev_token_last_char = last_char
        text = processEscapes(text, restore=True)
        yield text