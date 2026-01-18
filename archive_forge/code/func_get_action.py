import enum
def get_action(self, module):
    if self.matches(module.__name__):
        return Action.CONVERT
    return Action.NONE