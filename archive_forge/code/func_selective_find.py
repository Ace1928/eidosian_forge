import codecs
def selective_find(str, char, index, pos):
    """Return a pair (index, pos), indicating the next occurrence of
    char in str. index is the position of the character considering
    only ordinals up to and including char, and pos is the position in
    the full string. index/pos is the starting position in the full
    string."""
    l = len(str)
    while 1:
        pos += 1
        if pos == l:
            return (-1, -1)
        c = str[pos]
        if c == char:
            return (index + 1, pos)
        elif c < char:
            index += 1