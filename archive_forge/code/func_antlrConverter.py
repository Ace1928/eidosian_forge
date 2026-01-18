from pyparsing import (Word, ZeroOrMore, printables, Suppress, OneOrMore, Group,
def antlrConverter(antlrGrammarTree):
    pyparsingRules = {}
    antlrTokens = {}
    for antlrToken in antlrGrammarTree.tokens:
        antlrTokens[antlrToken.token_ref] = antlrToken.lit
    for antlrTokenName, antlrToken in list(antlrTokens.items()):
        pyparsingRules[antlrTokenName] = Literal(antlrToken)
    antlrRules = {}
    for antlrRule in antlrGrammarTree.rules:
        antlrRules[antlrRule.ruleName] = antlrRule
        pyparsingRules[antlrRule.ruleName] = Forward()
    for antlrRuleName, antlrRule in list(antlrRules.items()):
        pyparsingRule = __antlrRuleConverter(pyparsingRules, antlrRule)
        assert pyparsingRule != None
        pyparsingRules[antlrRuleName] <<= pyparsingRule
    return pyparsingRules