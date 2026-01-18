from io import StringIO
from typing import Callable
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNType import ATNType
from antlr4.atn.ATNState import *
from antlr4.atn.Transition import *
from antlr4.atn.LexerAction import *
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
def checkVersion(self):
    version = self.readInt()
    if version != SERIALIZED_VERSION:
        raise Exception('Could not deserialize ATN with version ' + str(version) + ' (expected ' + str(SERIALIZED_VERSION) + ').')