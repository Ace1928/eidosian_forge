from pyparsing import Suppress,Word,nums,alphas,alphanums,Combine,oneOf,\
def setBodyLength(tokens):
    strBody << Word(srange('[\\0x00-\\0xffff]'), exact=int(tokens[0]))
    return ''