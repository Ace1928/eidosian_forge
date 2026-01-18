from pyparsing import *
def makeRomanNumeral(n):

    def addDigit(n, limit, c, s):
        n -= limit
        s += c
        return (n, s)
    ret = ''
    while n >= 1000:
        n, ret = addDigit(n, 1000, 'M', ret)
    while n >= 900:
        n, ret = addDigit(n, 900, 'CM', ret)
    while n >= 500:
        n, ret = addDigit(n, 500, 'D', ret)
    while n >= 400:
        n, ret = addDigit(n, 400, 'CD', ret)
    while n >= 100:
        n, ret = addDigit(n, 100, 'C', ret)
    while n >= 90:
        n, ret = addDigit(n, 90, 'XC', ret)
    while n >= 50:
        n, ret = addDigit(n, 50, 'L', ret)
    while n >= 40:
        n, ret = addDigit(n, 40, 'XL', ret)
    while n >= 10:
        n, ret = addDigit(n, 10, 'X', ret)
    while n >= 9:
        n, ret = addDigit(n, 9, 'IX', ret)
    while n >= 5:
        n, ret = addDigit(n, 5, 'V', ret)
    while n >= 4:
        n, ret = addDigit(n, 4, 'IV', ret)
    while n >= 1:
        n, ret = addDigit(n, 1, 'I', ret)
    return ret