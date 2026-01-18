def skip_prologue(text, cursor):
    """skip any prologue found after cursor, return index of rest of text"""
    prologue_elements = ('!DOCTYPE', '?xml', '!--')
    done = None
    while done is None:
        openbracket = text.find('<', cursor)
        if openbracket < 0:
            break
        past = openbracket + 1
        found = None
        for e in prologue_elements:
            le = len(e)
            if text[past:past + le] == e:
                found = 1
                cursor = text.find('>', past)
                if cursor < 0:
                    raise ValueError("can't close prologue %r" % e)
                cursor = cursor + 1
        if found is None:
            done = 1
    return cursor