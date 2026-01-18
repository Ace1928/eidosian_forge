from copy import deepcopy
def references_table(self, table):
    return any((hasattr(part, 'references_table') and part.references_table(table) for part in self.parts.values()))