import re
@property
def rule_definition(self):
    if not self.rule or not self.definition:
        return None
    return self.definition.get(self.rule)