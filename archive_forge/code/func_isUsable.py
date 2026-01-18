from pyparsing import *
import random
import string
def isUsable(self, player, target):
    if self.usableConditionTest:
        return self.usableConditionTest(player, target)
    else:
        return False