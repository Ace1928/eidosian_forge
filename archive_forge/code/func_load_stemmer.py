import nltk
import os
import re
import itertools
import collections
import pkg_resources
@staticmethod
def load_stemmer(ensure_compatibility):
    """
        Load the stemmer that is going to be used if stemming is enabled
        Args
            ensure_compatibility: Use same stemmer and special "hacks" to product same results as in the official perl script (besides the number of sampling if not high enough)
        """
    Rouge.STEMMER = nltk.stem.porter.PorterStemmer('ORIGINAL_ALGORITHM') if ensure_compatibility else nltk.stem.porter.PorterStemmer()