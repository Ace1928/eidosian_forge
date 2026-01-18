def parse_subtags(subtags, expect=EXTLANG):
    """
    Parse everything that comes after the language tag: scripts, territories,
    variants, and assorted extensions.
    """
    if not subtags:
        return []
    subtag = subtags[0]
    tag_length = len(subtag)
    tagtype = None
    if tag_length == 1:
        if subtag.isalnum():
            return parse_extension(subtags)
        else:
            subtag_error(subtag)
    elif tag_length == 2:
        if subtag.isalpha():
            tagtype = TERRITORY
    elif tag_length == 3:
        if subtag.isalpha():
            if expect <= EXTLANG:
                return parse_extlang(subtags)
            else:
                order_error(subtag, EXTLANG, expect)
        elif subtag.isdigit():
            tagtype = TERRITORY
    elif tag_length == 4:
        if subtag.isalpha():
            tagtype = SCRIPT
        elif subtag[0].isdigit():
            tagtype = VARIANT
    else:
        tagtype = VARIANT
    if tagtype is None:
        subtag_error(subtag)
    elif tagtype < expect:
        order_error(subtag, tagtype, expect)
    else:
        if tagtype in (SCRIPT, TERRITORY):
            expect = tagtype + 1
        else:
            expect = tagtype
        typename = SUBTAG_TYPES[tagtype]
        if tagtype == SCRIPT:
            subtag = subtag.title()
        elif tagtype == TERRITORY:
            subtag = subtag.upper()
        return [(typename, subtag)] + parse_subtags(subtags[1:], expect)