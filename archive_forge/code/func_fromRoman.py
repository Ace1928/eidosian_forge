import re
def fromRoman(s):
    """convert Roman numeral to integer"""
    if not s:
        raise InvalidRomanNumeralError('Input can not be blank')
    if not romanNumeralPattern.search(s):
        raise InvalidRomanNumeralError('Invalid Roman numeral: %s' % s)
    result = 0
    index = 0
    for numeral, integer in romanNumeralMap:
        while s[index:index + len(numeral)] == numeral:
            result += integer
            index += len(numeral)
    return result