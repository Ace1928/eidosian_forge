def greedyWrap(inString, width=80):
    """
    Given a string and a column width, return a list of lines.

    Caveat: I'm use a stupid greedy word-wrapping
    algorythm.  I won't put two spaces at the end
    of a sentence.  I don't do full justification.
    And no, I've never even *heard* of hypenation.
    """
    outLines = []
    if inString.find('\n\n') >= 0:
        paragraphs = inString.split('\n\n')
        for para in paragraphs:
            outLines.extend(greedyWrap(para, width) + [''])
        return outLines
    inWords = inString.split()
    column = 0
    ptr_line = 0
    while inWords:
        column = column + len(inWords[ptr_line])
        ptr_line = ptr_line + 1
        if column > width:
            if ptr_line == 1:
                pass
            else:
                ptr_line = ptr_line - 1
            l, inWords = (inWords[0:ptr_line], inWords[ptr_line:])
            outLines.append(' '.join(l))
            ptr_line = 0
            column = 0
        elif not len(inWords) > ptr_line:
            outLines.append(' '.join(inWords))
            del inWords[:]
        else:
            column = column + 1
    return outLines