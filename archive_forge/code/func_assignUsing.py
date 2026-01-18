from pyparsing import Suppress,Word,nums,alphas,alphanums,Combine,oneOf,\
def assignUsing(s):

    def assignPA(tokens):
        if s in tokens:
            tokens[tokens[s]] = tokens[0]
            del tokens[s]
    return assignPA