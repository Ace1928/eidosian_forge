from pyparsing import Word, alphanums, Keyword, Group, Combine, Forward, Suppress, OneOrMore, oneOf
def evaluateNot(self, argument):
    return self.GetNot(self.evaluate(argument[0]))