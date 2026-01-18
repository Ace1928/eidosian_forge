import copy
def lemmas(self):
    """
        Returns a list of the lemmatized text of each token.

        Returns None if this annotation was not included.
        """
    if 'lemma' not in self.annotators:
        return None
    return [t[self.LEMMA] for t in self.data]