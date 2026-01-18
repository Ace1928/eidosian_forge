from pyparsing import alphas,nums, dblQuotedString, Combine, Word, Group, delimitedList, Suppress, removeQuotes
import string
def getLogLineBNF():
    global logLineBNF
    if logLineBNF is None:
        integer = Word(nums)
        ipAddress = delimitedList(integer, '.', combine=True)
        timeZoneOffset = Word('+-', nums)
        month = Word(string.ascii_uppercase, string.ascii_lowercase, exact=3)
        serverDateTime = Group(Suppress('[') + Combine(integer + '/' + month + '/' + integer + ':' + integer + ':' + integer + ':' + integer) + timeZoneOffset + Suppress(']'))
        logLineBNF = ipAddress.setResultsName('ipAddr') + Suppress('-') + ('-' | Word(alphas + nums + '@._')).setResultsName('auth') + serverDateTime.setResultsName('timestamp') + dblQuotedString.setResultsName('cmd').setParseAction(getCmdFields) + (integer | '-').setResultsName('statusCode') + (integer | '-').setResultsName('numBytesSent') + dblQuotedString.setResultsName('referrer').setParseAction(removeQuotes) + dblQuotedString.setResultsName('clientSfw').setParseAction(removeQuotes)
    return logLineBNF