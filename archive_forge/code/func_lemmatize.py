from nltk.corpus import wordnet as wn
def lemmatize(self, word: str, pos: str='n') -> str:
    """Lemmatize `word` using WordNet's built-in morphy function.
        Returns the input word unchanged if it cannot be found in WordNet.

        :param word: The input word to lemmatize.
        :type word: str
        :param pos: The Part Of Speech tag. Valid options are `"n"` for nouns,
            `"v"` for verbs, `"a"` for adjectives, `"r"` for adverbs and `"s"`
            for satellite adjectives.
        :param pos: str
        :return: The lemma of `word`, for the given `pos`.
        """
    lemmas = wn._morphy(word, pos)
    return min(lemmas, key=len) if lemmas else word