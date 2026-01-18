import codecs
def insertion_unsort(str, extended):
    """3.2 Insertion unsort coding"""
    oldchar = 128
    result = []
    oldindex = -1
    for c in extended:
        index = pos = -1
        char = ord(c)
        curlen = selective_len(str, char)
        delta = (curlen + 1) * (char - oldchar)
        while 1:
            index, pos = selective_find(str, c, index, pos)
            if index == -1:
                break
            delta += index - oldindex
            result.append(delta - 1)
            oldindex = index
            delta = 0
        oldchar = char
    return result