def stem_documents(self, docs):
    """Stem documents.

        Parameters
        ----------
        docs : list of str
            Input documents

        Returns
        -------
        list of str
            Stemmed documents.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.parsing.porter import PorterStemmer
            >>> p = PorterStemmer()
            >>> p.stem_documents(["Have a very nice weekend", "Have a very nice weekend"])
            ['have a veri nice weekend', 'have a veri nice weekend']

        """
    return [self.stem_sentence(x) for x in docs]