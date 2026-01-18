def remove_comments(s, collapse_spaces=True):
    """Removes any ()-style comments from a string.

    In HTTP, ()-comments can nest, and this function will correctly
    deal with that.

    If 'collapse_spaces' is True, then if there is any whitespace
    surrounding the comment, it will be replaced with a single space
    character.  Whitespace also collapses across multiple comment
    sequences, so that "a (b) (c) d" becomes just "a d".

    Otherwise, if 'collapse_spaces' is False then all whitespace which
    is outside any comments is left intact as-is.

    """
    if '(' not in s:
        return s
    A = []
    dostrip = False
    added_comment_space = False
    pos = 0
    if collapse_spaces:
        i = s.find('(')
        if i >= 0:
            while pos < i and s[pos] in LWS:
                pos += 1
            if pos != i:
                pos = 0
            else:
                dostrip = True
                added_comment_space = True
    while pos < len(s):
        if s[pos] == '(':
            _cmt, k = parse_comment(s, pos)
            pos += k
            if collapse_spaces:
                dostrip = True
                if not added_comment_space:
                    if len(A) > 0 and A[-1] and (A[-1][-1] in LWS):
                        A[-1] = A[-1].rstrip()
                        A.append(' ')
                        added_comment_space = True
        else:
            i = s.find('(', pos)
            if i == -1:
                if dostrip:
                    text = s[pos:].lstrip()
                    if s[pos] in LWS and (not added_comment_space):
                        A.append(' ')
                        added_comment_space = True
                else:
                    text = s[pos:]
                if text:
                    A.append(text)
                    dostrip = False
                    added_comment_space = False
                break
            else:
                if dostrip:
                    text = s[pos:i].lstrip()
                    if s[pos] in LWS and (not added_comment_space):
                        A.append(' ')
                        added_comment_space = True
                else:
                    text = s[pos:i]
                if text:
                    A.append(text)
                    dostrip = False
                    added_comment_space = False
                pos = i
    if dostrip and len(A) > 0 and A[-1] and (A[-1][-1] in LWS):
        A[-1] = A[-1].rstrip()
    return ''.join(A)