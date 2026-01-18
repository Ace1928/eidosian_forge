from pyparsing import ParserElement,LineEnd,Optional,Word,nums,Regex,\
@classmethod
def bailedTest(cls, num):
    ret = TAPTest(empty.parseString(''))
    ret.num = num
    ret.skipped = True
    return ret