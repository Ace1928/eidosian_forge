import re
def fontify(pytext, searchfrom=0, searchto=None):
    if searchto is None:
        searchto = len(pytext)
    search = matchRE.search
    idSearch = idRE.search
    tags = []
    tags_append = tags.append
    commentTag = 'comment'
    stringTag = 'string'
    keywordTag = 'keyword'
    identifierTag = 'identifier'
    start = 0
    end = searchfrom
    while 1:
        m = search(pytext, end)
        if m is None:
            break
        start = m.start()
        if start >= searchto:
            break
        match = m.group(0)
        end = start + len(match)
        c = match[0]
        if c not in '#\'"':
            if start != searchfrom:
                match = match[1:-1]
                start = start + 1
            else:
                match = match[:-1]
            end = end - 1
            tags_append((keywordTag, start, end, None))
            if match in ['def', 'class']:
                m = idSearch(pytext, end)
                if m is not None:
                    start = m.start()
                    if start == end:
                        match = m.group(0)
                        end = start + len(match)
                        tags_append((identifierTag, start, end, None))
        elif c == '#':
            tags_append((commentTag, start, end, None))
        else:
            tags_append((stringTag, start, end, None))
    return tags