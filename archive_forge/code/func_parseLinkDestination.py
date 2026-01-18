from ..common.utils import charCodeAt, unescapeAll
def parseLinkDestination(string: str, pos: int, maximum: int) -> _Result:
    lines = 0
    start = pos
    result = _Result()
    if charCodeAt(string, pos) == 60:
        pos += 1
        while pos < maximum:
            code = charCodeAt(string, pos)
            if code == 10:
                return result
            if code == 60:
                return result
            if code == 62:
                result.pos = pos + 1
                result.str = unescapeAll(string[start + 1:pos])
                result.ok = True
                return result
            if code == 92 and pos + 1 < maximum:
                pos += 2
                continue
            pos += 1
        return result
    level = 0
    while pos < maximum:
        code = charCodeAt(string, pos)
        if code is None or code == 32:
            break
        if code < 32 or code == 127:
            break
        if code == 92 and pos + 1 < maximum:
            if charCodeAt(string, pos + 1) == 32:
                break
            pos += 2
            continue
        if code == 40:
            level += 1
            if level > 32:
                return result
        if code == 41:
            if level == 0:
                break
            level -= 1
        pos += 1
    if start == pos:
        return result
    if level != 0:
        return result
    result.str = unescapeAll(string[start:pos])
    result.lines = lines
    result.pos = pos
    result.ok = True
    return result