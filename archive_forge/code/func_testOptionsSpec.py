import unittest
from . import antlr_grammar
def testOptionsSpec(self):
    text = 'options {\n                            language = Python;\n                        }'
    antlr_grammar.optionsSpec.parseString(text)