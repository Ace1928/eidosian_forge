import unittest
from . import antlr_grammar
def testGrammar(self):
    text = "grammar SimpleCalc;\n\noptions {\n    language = Python;\n}\n\ntokens {\n    PLUS     = '+' ;\n    MINUS    = '-' ;\n    MULT    = '*' ;\n    DIV    = '/' ;\n}\n\n/*------------------------------------------------------------------\n * PARSER RULES\n *------------------------------------------------------------------*/\n\nexpr    : term ( ( PLUS | MINUS )  term )* ;\n\nterm    : factor ( ( MULT | DIV ) factor )* ;\n\nfactor    : NUMBER ;\n\n\n/*------------------------------------------------------------------\n * LEXER RULES\n *------------------------------------------------------------------*/\n\nNUMBER    : (DIGIT)+ ;\n\n/* WHITESPACE : ( '\t' | ' ' | '\r' | '\n'| '\x0c' )+     { $channel = HIDDEN; } ; */\n\nfragment DIGIT    : '0'..'9' ;"
    antlrGrammarTree = antlr_grammar.grammarDef.parseString(text)
    pyparsingRules = antlr_grammar.antlrConverter(antlrGrammarTree)
    pyparsingRule = pyparsingRules['expr']
    pyparsingTree = pyparsingRule.parseString('2 - 5 * 42 + 7 / 25')
    pyparsingTreeList = pyparsingTree.asList()
    print(pyparsingTreeList)
    self.assertEqual(pyparsingTreeList, [[[['2'], []], [['-', [['5'], [['*', ['4', '2']]]]], ['+', [['7'], [['/', ['2', '5']]]]]]]])