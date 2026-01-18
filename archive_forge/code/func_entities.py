import copy
def entities(self):
    """
        Returns a list of named-entity-recognition tags of each token.

        Returns None if this annotation was not included.
        """
    if 'ner' not in self.annotators:
        return None
    return [t[self.NER] for t in self.data]